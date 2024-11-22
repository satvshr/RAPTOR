from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from lm_studio import LMStudioLLM
from translation import translation_template
from routing import routing_template
from indexing import indexing_template
from retrieval import retrieval_template
from generation import generation_template

from pdf_summarizer import get_summaries

load_dotenv()

lm_studio_llm = LMStudioLLM(path='complsetions')

llm_chain_file = RunnableSequence(
    translation_template() | 
    lm_studio_llm | 
    (lambda output: routing_template().format(
        translated_questions=output,  
        file_summaries=file_summaries  
    )) | 
    lm_studio_llm 
    # indexing_template() | 
    # lm_studio_llm | 
    # retrieval_template() | 
    # lm_studio_llm | 
    # generation_template()
)

llm_chain_no_file = RunnableSequence(
    translation_template() | lm_studio_llm
)

question = {'question': input("Enter q: ")}
files = []

for i in range(int(input("Enter the number of files: "))):
    files.append(input("Enter file name: "))
        
if len(files) == 0:
    answer = llm_chain_no_file.invoke(question)
else:
    # summaries in the format [[file_name, summary], ..]
    summaries = get_summaries(files)
    file_summaries = "\n".join([f"{file}: {summary}" for file, summary in summaries])

    answer = llm_chain_file.invoke({
        'question': question,
        'file_summaries': file_summaries
    })

print(answer) 