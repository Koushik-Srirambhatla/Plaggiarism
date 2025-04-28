import os
import re
import requests
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('all-MiniLM-L6-v2')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif ext == 'docx':
        doc = docx.Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def compare_texts(text1, text2, threshold=0.6):
    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)
    if not sents1 or not sents2:
        return []
    embeddings1 = model.encode(sents1, convert_to_tensor=True)
    embeddings2 = model.encode(sents2, convert_to_tensor=True)

    results = []
    for i, emb1 in enumerate(embeddings1):
        scores = util.cos_sim(emb1, embeddings2)[0]
        max_score = max(scores)
        if max_score >= threshold:
            results.append((sents1[i], float(max_score)))
    return results

def search_google_snippet(query):
    params = {
        "q": query,
        "api_key": "0bfdb56d51fdcc082ff90bf916b2037e4ab31b02875b2dfe1d355e4920d7c414",
        "engine": "google",
        "num": "3"
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        urls = [res['link'] for res in data.get("organic_results", [])[:3]]
        return urls
    except:
        return []

def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except:
        return ""

def online_check(input_text, threshold=0.6):
    sents = split_sentences(input_text)
    results = []

    def process_sentence(sent):
        urls = search_google_snippet(sent)
        matches = []
        for url in urls:
            web_text = extract_text_from_url(url)
            if not web_text:
                continue
            comparison = compare_texts(sent, web_text, threshold)
            for line, score in comparison:
                # Include snippet of the web text
                snippet = web_text[:300]  # limit snippet length
                matches.append((line, score, url, snippet))
        return matches

    with ThreadPoolExecutor() as executor:
        all_results = executor.map(process_sentence, sents)
        for res in all_results:
            results.extend(res)
    return results

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload_text', methods=['POST'])
def upload_text():
    user_text = request.form.get('userText')
    if user_text:
        filename = "user_input.txt"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(user_text)
        
        # Now do plagiarism check same as file upload
        input_text = extract_text(filepath)
        plagiarized_sentences = []

        for f_name in os.listdir(UPLOAD_FOLDER):
            other_path = os.path.join(UPLOAD_FOLDER, f_name)
            if other_path != filepath:
                other_text = extract_text(other_path)
                results = compare_texts(input_text, other_text)
                plagiarized_sentences.extend([(s, sc, "Local File", "") for s, sc in results])

        # Online check
        online_results = online_check(input_text)
        plagiarized_sentences.extend(online_results)

        return render_template('result.html', results=plagiarized_sentences, filename=filename)
    
    return redirect('/')

def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            input_text = extract_text(filepath)
            plagiarized_sentences = []

            # Compare with all other uploaded docs
            for f in os.listdir(UPLOAD_FOLDER):
                other_path = os.path.join(UPLOAD_FOLDER, f)
                if other_path != filepath:
                    other_text = extract_text(other_path)
                    results = compare_texts(input_text, other_text)
                    plagiarized_sentences.extend([(s, sc, "Local File", "") for s, sc in results])

            # Online check
            online_results = online_check(input_text)
            plagiarized_sentences.extend(online_results)

            return render_template('result.html', results=plagiarized_sentences, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)