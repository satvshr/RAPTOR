from langsmith import Client as traceable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
import ast, re
from langchain.schema import Document
from langchain_community.embeddings import GPT4AllEmbeddings
from utils.find_documents import find_documents

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 200,
    chunk_overlap = 50
)

top_k = 7  # Maximum splits to retrieve to answer the question

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join([page.extract_text() for page in reader.pages])

# Input in the format [file names, ...]
def split_documents(documents):
    # List to store splits of all documents
    splits = []

    for doc in documents:
        # Extract text from the PDF file
        pdf_text = extract_text_from_pdf(rf"C:\Users\satvm\Downloads\{doc}.pdf")
        
        # Wrap the text in a document object
        document = Document(page_content=pdf_text)

        # Split the document into chunks
        local_splits = text_splitter.split_documents([document])  # Pass as a list
        splits.extend(local_splits)

    return splits  # Already flattened

def extract_questions(text):
    # Match all substrings ending with a question mark
    questions = re.findall(r'[^?]*\?+', text)
    # Strip leading/trailing whitespace from each question
    return [q.strip() for q in questions]

def get_unique_splits(splits):
    # Flatten list of splits from all questions
    unique_splits = {}
    for split in splits:
        # Use `page_content` as the key to ensure uniqueness
        unique_splits[split.page_content] = split
    return list(unique_splits.values())


# @traceable
def indexing_template():
    def process_questions_and_documents(documents, questions):
        # Get the list of files from the output in the form of a string
        documents = re.search(r'\[.*?\]', documents).group()
        splits = split_documents(ast.literal_eval(documents))  # Convert string list to actual list

        # Create a Chroma vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())
        retriever = vectorstore.as_retriever()

        # Define the retrieval chain
        def retrieval_chain(questions):
            questions = extract_questions(questions)
            sorted_docs = find_documents(retriever, questions)
            return get_unique_splits(sorted_docs)[:top_k]

        # Invoke the chain with the questions
        results = retrieval_chain(questions)
        print(len(results))
        return results

    return process_questions_and_documents