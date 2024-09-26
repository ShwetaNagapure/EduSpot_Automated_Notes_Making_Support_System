from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

with app.app_context():
    db.create_all()

def split_audio_chunks(audio_path, min_silence_len=1000, silence_thresh=-40, keep_silence=500):
    sound = AudioSegment.from_file(audio_path)
    audio_chunks = split_on_silence(sound, 
                                    min_silence_len=min_silence_len,
                                    silence_thresh=silence_thresh,
                                    keep_silence=keep_silence)
    return audio_chunks

def transcribe_audio_chunks(chunks, recognizer, language='en-US'):
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        chunk_audio = io.BytesIO()
        chunk.export(chunk_audio, format="wav")
        chunk_audio.seek(0)
        with sr.AudioFile(chunk_audio) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language=language)
                full_transcription += text + " "
            except sr.UnknownValueError:
                print(f"Chunk {i}: Could not understand audio")
            except sr.RequestError as e:
                print(f"Chunk {i}: Could not request results; {e}")
    return full_transcription

def transcribe_large_audio(file_path):
    recognizer = sr.Recognizer()
    audio_chunks = split_audio_chunks(file_path)
    transcription = transcribe_audio_chunks(audio_chunks, recognizer)
    return transcription

def create_vector_store_and_retriever(transcription_text):
    text_documents = [{"page_content": transcription_text}]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc = Document(page_content=text_documents[0]["page_content"])
    docs = text_splitter.split_documents([doc])

    HF_TOKEN = "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"
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")
    
    vectorstore = Chroma.from_documents(docs, embeddings)
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def generate_notes(transcription_text):
    groq_api_key = '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'
    model_name = "mixtral-8x7b-32768"
    groq = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    prompt_template = f"""
    Task: Summarize the following document into concise, detailed, and structured notes.
    Instructions:
    1. Read the provided document carefully.
    2. Extract key concepts, important data points, and actionable insights with detailed explanations and context from the document.
    3. Format the notes with headings, subheadings, and bullet points where appropriate.
    4. Ensure clarity, depth, and brevity.
    5. Use the following structure for the notes:

    Notes Format:

    ### 1. Title of the Document
    *(Inferred title from the content)*

    ### 2. Key Concepts
    *(Summarize the main ideas or themes covered with detailed context from the document)*  
    - [Concept 1]: Provide a detailed explanation of the concept, mentioning any relevant examples or scenarios from the text.  
    - [Concept 2]: Provide a detailed explanation of the concept, linking back to specific parts of the document.

    ### 3. Important Data Points
    *(Highlight significant figures, dates, or statistics with detailed explanations of their relevance or importance)*  
    - Data point 1: Include a detailed description and explain why this point is important in the context of the document.  
    - Data point 2: Provide further explanation with relevant details from the text.

    ### 4. Actionable Insights
    *(Outline any key actions, recommendations, or steps suggested by the document, with more context and reasoning)*  
    - Insight 1: Provide more detailed reasoning or suggestions based on the document's content.  
    - Insight 2: Expand on the insight by mentioning how it connects to the main themes of the document.

    ### 5. Conclusion/Next Steps
    *(Summarize the final takeaway or suggested next steps, offering more detailed conclusions based on the text)*

    Document Content:
    {transcription_text}
    """
    response = groq.invoke(prompt_template)
    return response.content

def create_chatbot_chain(ensemble_retriever):
    llm = ChatGroq(groq_api_key='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',
                   model_name="mixtral-8x7b-32768")

    template = """
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>
    Question: {input}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = (
        {"context": ensemble_retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    
    return chain

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            transcription = transcribe_large_audio(file_path)
            notes = generate_notes(transcription)
            ensemble_retriever = create_vector_store_and_retriever(transcription)
            session['ensemble_retriever'] = ensemble_retriever
            session['notes'] = notes
            session['transcription'] = transcription
            return redirect(url_for('chat'))
    return render_template('upload.html')
@app.route('/notes')
def view_notes():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    notes = session.get('notes', '')  # Retrieve the notes from the session
    return render_template('notes.html', notes=notes) 
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'username' not in session or 'ensemble_retriever' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        user_input = request.form['user_input']
        chain = create_chatbot_chain(session['ensemble_retriever'])
        response = chain.invoke(user_input)
        return jsonify({'response': response})
   
    return render_template('chat.html', notes=session.get('notes', ''), transcript=session.get('transcription', ''))

if __name__ == '__main__':
    app.run(debug=True)