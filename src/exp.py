from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from utils.lm_studio import LMStudioLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings

from src.translation import translation_template
from src.routing import logical_routing_template, semantic_routing_template
from src.indexing import indexing_template
from src.raptor import raptor_template
from src.retrieval import retrieval_template
from src.generation import generation_template

from utils.pdf_summarizer import get_summaries

# Load environment variables
load_dotenv()

# Initialize LLM, number of splits to retrieve, text splitter, and the embedder
lm_studio_llm = LMStudioLLM(path='completions')
top_k_indexing = 30
top_k_raptor = 5
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap = 50
)
embedder = GPT4AllEmbeddings()

# Input question
question = {'question': "What is twitter sentiment analysis?"}

# Precompute translation_output
translation_result = translation_template() | lm_studio_llm
translation_output = translation_result.invoke(question)

# Gather files
files = ["1", "2", "3"]
# for i in range(int(input("Enter the number of files: "))):
#     files.append(input("Enter file name: "))

# Generate file summaries
file_summaries = ""
if len(files) > 0:
    # summaries in the format [[file_name, summary], ..]
    summaries = get_summaries(files)
    file_summaries = "\n".join([f"{file}: {summary}" for file, summary in summaries])

# Define llm_chain_file
llm_chain_file = RunnableSequence(
    # Logical routing
    # (lambda question: logical_routing_template().format(
    #     question=question['question'],  
    #     file_summaries=file_summaries  
    # )) | 

    # Semantic routing
    (lambda: semantic_routing_template().format(
        questions=translation_output,
        file_summaries=file_summaries,
        embedder=embedder
    )) |
    lm_studio_llm | 
    (lambda doc_name_list: indexing_template()(
        documents=doc_name_list,
        questions=translation_output,
        text_splitter=text_splitter,
        embedder=embedder,
        top_k=top_k_indexing
    )) |
    (lambda splits_list: raptor_template()(
        doc_splits=splits_list,
        embedder=embedder,
        top_k=top_k_raptor
    ))
)

# Define llm_chain_no_file
llm_chain_no_file = RunnableSequence(
    translation_template() | lm_studio_llm
)

# Execute chain based on files
if len(files) == 0:
    answer = llm_chain_no_file.invoke(question)
else:
    answer = llm_chain_file.invoke({
        'question': question,
        'file_summaries': file_summaries
    })

print(answer)