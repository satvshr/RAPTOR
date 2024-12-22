from utils.find_documents import find_documents
from src.indexing import extract_questions

def retrieval_template():
    def get_best_nodes(cluster_summaries, questions, data_point_to_clusters):
        questions = extract_questions(questions)
        # Get the cluster with the highest cosine similarity
        best_cluster = cluster_summaries.index(find_documents(cluster_summaries, questions, embedder)[0])
        # Get the nodes for the best cluster
        best_cluster_nodes = data_point_to_clusters[best_cluster]

    return get_best_nodes