from langchain.prompts import PromptTemplate

def generation_template():
    print("Generating the answer to the user question using the information obtained...\n")
    return PromptTemplate(
        input_variables=["question", "best_nodes"],
        template="""
        You are a highly intelligent assistant aiding with complex query resolution. 
        Consider the following question and the provided key points for context.

        Question:
        {question}

        Key Points:
        {best_nodes}

        Generate a comprehensive and accurate response based on this information. Ensure the response directly addresses the question while integrating the key points provided.

        Response:
        """
)