from src.indexing import extract_questions
from utils.gmm import gmm, get_optimal_clusters
from utils.umap import umap
from .raptor import get_summaries
from utils.pdf_summarizer import query
from utils.find_documents import find_documents
from .indexing import get_unique_splits
from langchain_community.embeddings import GPT4AllEmbeddings
import numpy as np

def retrieval_template():
    def get_best_nodes(best_cluster_nodes, questions, embedder, top_k=4, reduced_dim=10, threshold=0.1):
        questions = extract_questions(questions)
        # list to summarize nodes of each level only
        current_lvl = best_cluster_nodes.copy()

        # Build a bottom-up tree and use the collapsed tree approach to ensure best performance
        n_clusters = 0
        while True:
            # A lot of the code is the same as raptor.py as we are using the same process
            n_neighbors = int((len(current_lvl) - 1) ** 0.5)

            # Perform dimensionality reduction
            lower_dim = umap(current_lvl, embedder, n_neighbors, reduced_dim)

            # Get the number of clusters which would give the ebst performance
            n_clusters = get_optimal_clusters(lower_dim, reduced_dim)
            if n_clusters == 1: # Break the loop if the optimal cluster is 1
                break

            responsibilities, _, _, _, _ = gmm(lower_dim, n_clusters)

            # Implement soft clustering by using a threshold
            labels = [np.where(prob > threshold)[0] for prob in responsibilities]

            # Create a mapping of cluster to data points in it
            data_point_to_clusters = {i: label.tolist() for i, label in enumerate(labels)}

            # Get summaries of all the clusters
            cluster_summaries = get_summaries(current_lvl, data_point_to_clusters)

            # Update variable to contain the summaries of only the current level
            current_lvl = cluster_summaries

            # Accumalate all nodes of the entire tree under one level (collapsed tree approach) and append it to a list
            best_cluster_nodes.extend(current_lvl)

        # Once the number of optimal clusters is 1, get summaries of the nodes in it and append it to the list
        # This is the root node
        summary = "\n".join(node for node in current_lvl)
        while True:
            try:
                summary = query({"inputs": summary})[0]['summary_text']
                break
            except KeyError:
                print("Error: The hugging face summarizer API threw an error")
        best_cluster_nodes.append(summary)

        print("Total length of all the nodes in the cluster: ", len(best_cluster_nodes))
        # Get the unique nodes arranged in descending order of cosine similarity
        nodes = find_documents(best_cluster_nodes, questions, embedder)
        # Return top_k nodes based on preference
        unique_nodes = get_unique_splits(nodes)[:top_k]

        # Return top_k nodes as a string seperated by the new line character 
        return "\n".join(node for node in unique_nodes)

    return get_best_nodes