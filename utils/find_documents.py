import numpy as np

def cosine_similarity_search(vectorstore, embedded_question, embedder):
    # List to store cosine similarity scores in the (doc, similarity) format
    doc_cosine_pair = []
    try:
        documents = vectorstore.get()['documents']
    except AttributeError:
        # If vectorstore does not have a 'get' method, treat it as a list
        documents = vectorstore

    # Calculate cosine similarity for each document
    for doc in documents:
        embedded_doc = embedder.embed_query(doc)
        numerator = np.dot(embedded_doc, embedded_question)
        denominator = np.linalg.norm(embedded_doc) * np.linalg.norm(embedded_question)
        if denominator == 0:  # Avoid division by zero
            cosine = 0.0
        else:
            cosine = numerator / denominator
        doc_cosine_pair.append((doc, cosine))
    
    return doc_cosine_pair

def find_documents(vectorstore, questions, embedder):
    # Universal list for all doc-cosine pairs for all questions
    doc_cosine_pairs = []
    for question in questions:
        embedded_question = embedder.embed_query(question)
        doc_cosine_pair_per_q = cosine_similarity_search(vectorstore, embedded_question, embedder)
        doc_cosine_pairs.extend(doc_cosine_pair_per_q)

    # Sort by similarity score in descending order
    doc_cosine_pairs = sorted(doc_cosine_pairs, key=lambda x: x[1], reverse=True)
    sorted_docs = [doc for doc, _ in doc_cosine_pairs]

    return sorted_docs