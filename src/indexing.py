from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
import ast, re, os
from dotenv import load_dotenv
from langchain.schema import Document
from utils.find_documents import find_documents

load_dotenv()

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join([page.extract_text() for page in reader.pages])

# Input in the format [file names, ...]
def split_documents(documents, text_splitter):
    # List to store splits of all documents
    splits = []

    for doc in documents:
        # Extract text from the PDF file
        pdf_text = extract_text_from_pdf(rf"{os.getenv('FILE_PATH')}\{doc}.pdf")
        
        # Wrap the text in a document object
        document = Document(page_content=pdf_text)

        # Split the document into chunks
        local_splits = text_splitter.split_documents([document])  # Pass as a list
        splits.extend(local_splits)

    return splits  # Already flattened

def extract_questions(text):
    # Match all substrings ending with a question mark
    questions = re.findall(r'(?:^\d+\.\s*)?(.*?\?+)', text, re.MULTILINE)
    # Strip leading/trailing whitespace from each question
    return [q.strip() for q in questions if q.strip()]

def get_unique_splits(splits):
    seen = set()
    unique_splits = []
    for split in splits:
        if split not in seen:
            unique_splits.append(split)
            seen.add(split)
    return unique_splits

def indexing_template():
    def process_questions_and_documents(documents, questions, text_splitter, embedder, top_k):
        print("\nPerforming indexing...")
        # Get the list of files from the output in the form of a string (for logical routing)
        if isinstance(documents, str):
            documents = re.search(r'\[.*?\]', documents).group()
            splits = split_documents(ast.literal_eval(documents), text_splitter)  # Convert string list to actual list
            
        splits = split_documents(documents, text_splitter)
        # Create a Chroma vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)

        questions = extract_questions(questions)
        sorted_docs = find_documents(vectorstore, questions, embedder)
        results = get_unique_splits(sorted_docs)[:top_k]

        print("Finished indexing.")
        print(f"Obtained the {top_k} unique document splits best matching our question using cosine similarity search.\n")
        # Results contains the top_k unique document splits that best match the questions
        return results

    return process_questions_and_documents