from langsmith import Client as traceable
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from lm_studio import LMStudioLLM
from PyPDF2 import PdfReader
import ast, re
from langchain.schema import Document
from langchain_community.embeddings import GPT4AllEmbeddings

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
        print(documents)
        splits = split_documents(ast.literal_eval(documents))  # Convert string list to actual list
        print(splits)

        # Create a Chroma vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())
        retriever = vectorstore.as_retriever()

        # Define the retrieval chain
        def retrieval_chain(questions):
            retrieved_docs = []
            for question in questions:
                docs = retriever.invoke(question)
                retrieved_docs.extend(docs)
            return get_unique_splits(retrieved_docs)

        # Invoke the chain with the questions
        results = retrieval_chain(questions)
        return results

    return process_questions_and_documents

# indexing_template()(
#         documents='["1", "2"]', 
#         questions="""
# 1: [{'summary_text': 'The sentiment analysis task has been performed by collecting the dataset from the publically available sources. A new reliable dataset is then subject to various pre -processing techniques and then the feature extraction techniques. The results are passed to the deep learning technique s out of which global vectors ( glovec) have the highest accu racy of 75%.'}]
# 2: [{'summary_text': 'The contemporary work is done as slice of the shared task inSentiment Analysis in Indian Languages (SAIL 2015), constrained vari-ety. Social media allows people to create and share or exchange opinionsbased on many perspectives. A supervised algorithm is used for clas-sifying the tweets into positive, negative and neutral labels.'}]
# 3: [{'summary_text': 'This system holds an edge over the current rating system of star values by providing the users with a more precise and descriptive result. The main disadvantage of thestar system is that it does not provide enough choice to the user. The methodology in this paper, named ARAS or Automated Review Analyzing System,overcomes this issue by using sentiment analysis.'}]
# """
#     )