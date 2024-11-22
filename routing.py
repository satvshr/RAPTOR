from langsmith import Client as traceable
from langchain.prompts import PromptTemplate

# @traceable
def routing_template():
    return PromptTemplate(
        input_variables=["translated_questions", "file_summaries"],
        template="""
Given a list of file summaries and a user question, determine which file/files is the best match for answering the question. 

User Question: {translated_questions}

File Summaries: 
{file_summaries}

Choose the files that most closely aligns with the user question and explain why it is the best match.
Response:
"""
)