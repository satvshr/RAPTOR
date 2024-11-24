from langsmith import Client as traceable
from langchain.prompts import PromptTemplate

# @traceable
def routing_template():
    return PromptTemplate(
        input_variables=["question", "file_summaries"],
        template="""
        Given a list of file summaries and a user question, identify which files best match the question.

        User Question: {question}

        File Summaries: 
        {file_summaries}

        Respond with only the names of the matching files as a string formatted like a Python list, nothing else.

        Response:
        """
)