from utils.umap import umap
from utils.gmm import gmm, get_optimal_clusters
import numpy as np
from utils.pdf_summarizer import query
from .indexing import extract_questions
from utils.find_documents import find_documents

def get_summaries(doc_splits, data_point_to_clusters, isLayer=False):
    # Contains the summary of each cluster
    summaries = []
    MAX_TOKENS = 3000

    for i in data_point_to_clusters:
        # Flag to check if summaries are being formed for a cluster or for the nodes in the tree
        if not isLayer:
            print(f"Generating summaries for cluster {i+1}...")

        # Summary contains all nodes seperated by a new line
        summary = "\n".join(doc_splits[node] for node in data_point_to_clusters[i])
        n_summary_bits = len(summary) // MAX_TOKENS

        # loop to summarize a clusters summary using looping
        if n_summary_bits > 0:
            for i in range(n_summary_bits):
                bits_summaries = []
                bits_summaries.append(query({"inputs": summary[:MAX_TOKENS]}))
                summary = summary[MAX_TOKENS:]

            bits_summaries.append(query({"inputs": summary})) # For the remaining tokens whose len < MAX_TOKENS
            # Aggregate all the summaries of all the nodes in the cluster
            summary = "\n".join(bits[0]['summary_text'] for bits in bits_summaries if isinstance(bits, list)) # Ignore any node which was not summarized because of an API error
        
        # Get the summary of all the nodes' summaries
        summary = query({"inputs": summary})[0]['summary_text']
        summaries.append(summary)

    return summaries

def raptor_template():
    def raptor(doc_splits, questions, embedder, reduced_dim=10, threshold=0.1):
        # Calculate n_neighbors to consider while performing clustering
        n_neighbors = int((len(doc_splits) - 1) ** 0.5)
        print(f"For UMAP the nearest {n_neighbors} neighbors are used and dimensionality is reduced to {reduced_dim}.\n")
        print("Performing UMAP...")
        # Perform dimensionality reduction
        lower_dim = umap(doc_splits, embedder, n_neighbors, reduced_dim)

        # Get the number of clusters which would give the best performance
        print("Obtaining the optimal number of clusters which will perform the best...")
        n_clusters = get_optimal_clusters(lower_dim, reduced_dim)
        print(f"The number of clusters which will be used for GMM is {n_clusters}, which was determined using the BIC algorithm.\n")
        responsibilities, _, _, _, _ = gmm(lower_dim, n_clusters)

        # Implement soft clustering by using a threshold
        labels = [np.where(prob > threshold)[0] for prob in responsibilities]

        # Create a mapping of cluster to data points in it
        data_point_to_clusters = {i: label.tolist() for i, label in enumerate(labels)}

        # Ensure each cluster has at least one data point
        for i in range(len(data_point_to_clusters)):
            if len(data_point_to_clusters[i]) == 0:
                # Find the data point with the highest probability for this cluster
                highest_prob_idx = np.argmax(responsibilities[i, :])
                data_point_to_clusters[i].append(highest_prob_idx)

        # Get summaries of all the clusters
        print("Generating sumaries for all the clusters:\n")
        cluster_summaries = get_summaries(doc_splits, data_point_to_clusters)

        # Get the cluster with the highest cosine similarity
        questions = extract_questions(questions)
        best_cluster = cluster_summaries.index(find_documents(cluster_summaries, questions, embedder)[0])
        print("Selected a cluster which best matches our question using cosine similarity search.")

        # Get the nodes for the best cluster
        best_cluster_nodes = [doc_splits[idx] for idx in data_point_to_clusters[best_cluster]]

        return best_cluster_nodes
    
    return raptor