from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from utils.lm_studio import LMStudioLLM
from src.translation import translation_template
from src.routing import routing_template
from src.indexing import indexing_template
from src.retrieval import retrieval_template
from src.generation import generation_template

from utils.pdf_summarizer import get_summaries

load_dotenv()

lm_studio_llm = LMStudioLLM(path='completions')

llm_chain_file = RunnableSequence(
    translation_template() | 
    lm_studio_llm |
    (lambda translation_output: {
        'routing_output': routing_template().format(
            question=question['question'],  
            file_summaries=file_summaries  
        ),
        'translation_output': translation_output
    }) | 
    lm_studio_llm | 
    (lambda outputs: indexing_template().format(
        routing_output=outputs['routing_output'], 
        translation_output=outputs['translation_output']
    )) | 
    lm_studio_llm
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