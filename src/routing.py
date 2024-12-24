from langchain.prompts import PromptTemplate
import numpy as np
from .indexing import extract_questions

# Logical routing
def logical_routing_template():
    return PromptTemplate(
        input_variables=["question", "file_summaries"],
        template="""
        Given a list of file summaries and a user question, identify which files best match the question and output only a single string.

        User Question: {question}

        File Summaries: 
        {file_summaries}

        Respond in a single line with only the names of the matching files as a string formatted like a Python list, nothing else, only 1 single sentence.

        Response:
        """
)

# Semantic routing
def semantic_routing_template():
    def choose_files(questions, file_summaries, embedder, threshold=0.35):
        print("Performing semantic routing...")
        # summaries in the format [[file_name, summary], ..]
        related_files = []
        questions = extract_questions(questions)
        for question in questions:
            embedded_question = embedder.embed_query(question)
            for file in file_summaries:
                content = file[1][0]['summary_text']
                cosine = np.dot(embedder.embed_query(content), embedded_question) / np.linalg.norm(embedder.embed_query(content)) * np.linalg.norm(embedded_question)
                if cosine > threshold:
                    if file[0] not in related_files:
                        related_files.append(file[0])
                        
        print("Finished routing.")
        print("The chosen files after routing are:")
        for i in related_files:
            print(i)

        return related_files
        
    return choose_files