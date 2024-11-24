from langsmith import Client as traceable
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from lm_studio import LMStudioLLM
from PyPDF2 import PdfReader
import ast, re
from langchain.schema import Document

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap=50
)

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

def get_unique_splits(splits):
    # flatten list of splits from all questions
    flattened_splits = [split for split in splits]
    # Get unique splits
    unqiue_splits = list(set(flattened_splits))
    return unqiue_splits

# @traceable
def indexing_template():
    def process_questions_and_documents(documents, questions):
        documents = re.search(r'\[.*?\]', documents).group()
        print(documents)
        splits = split_documents(ast.literal_eval(documents)) # Convert the string representation of list to a list
        vectorstore = Chroma.from_documents(documents=splits, embedding=LMStudioLLM(path='embeddings'))
        retriever = vectorstore.as_retriever()

        retrieval_chain = questions | retriever.map() | get_unique_splits
        return retrieval_chain.invoke()
    
    return process_questions_and_documents