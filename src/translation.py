from langchain.prompts import PromptTemplate

def translation_template():
    return PromptTemplate(
        input_variables=["question"],
        template="""Given the following user question, generate three similar variations of the question without changing its core meaning. Each variation should be in a single line, without any additional explanation or labels. The response should only consist of three distinct variations of the question.

User Question: {question}

Response:"""
)