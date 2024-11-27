from langchain_community.embeddings import GPT4AllEmbeddings
import numpy as np

def cosine_similarity_search(retriever, embedded_question):
    # List to store cosine similarity scores in the (doc, similarity) format
    doc_cosine_pair = []

    # Calculate cosine similarity for each document
    for doc in retriever:
        print(doc)
        numerator = np.dot(doc, embedded_question)
        denominator = np.linalg.norm(doc) * np.linalg.norm(embedded_question)
        if denominator == 0:  # Avoid division by zero
            cosine = 0.0
        else:
            cosine = numerator / denominator
        doc_cosine_pair.append((doc, cosine))
    
    return doc_cosine_pair

def find_documents(retriever, questions):
    # Universal list for all questions
    doc_cosine_pairs = []
    for question in questions:
        embedded_question = GPT4AllEmbeddings().embed_query(question)
        doc_cosine_pair_per_q = cosine_similarity_search(retriever, embedded_question)
        doc_cosine_pairs.extend(doc_cosine_pair_per_q)

    # Sort by similarity score in descending order
    doc_cosine_pairs = sorted(doc_cosine_pairs, key=lambda x: x[1], reverse=True)
    sorted_docs = [doc for _, doc in doc_cosine_pairs]
    print(sorted_docs)
    return sorted_docs