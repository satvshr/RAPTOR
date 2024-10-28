import PyPDF2
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Token limiter so that we don't send too many tokens and get thrown an error message
MAX_TOKENS = 3500
summarizer = os.getenv('SUMMARIZER_API_KEY')
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {summarizer}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}", "message": response.text}
    
# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text

# Send the text to a summarizer
def summarizer(file):
    pdf_path = rf"C:\Users\satvm\Downloads\{file}.pdf"
    document_text = extract_text_from_pdf(pdf_path)[:MAX_TOKENS]
    summary = query({"inputs": document_text})
    return [file, summary]

# Get summaries of all files
def get_summaries(files):
    summaries = []
    for i in range(len(files)):
        summaries.append(summarizer(files[i]))
    
    return summaries
