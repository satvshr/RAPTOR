from utils.umap import umap
from utils.gmm import gmm, get_optimal_clusters
import numpy as np
from utils.pdf_summarizer import query

def get_summaries(doc_splits, data_point_to_clusters):
    # Contains the summary of each cluster
    summaries = []
    MAX_TOKENS = 3000

    for i in data_point_to_clusters:
        # Summary conains all nodes seperated by a new line
        summary = "\n".join(doc_splits[node] for node in data_point_to_clusters[i])
        print("len summary ", len(summary))
        n_summary_bits = len(summary) // MAX_TOKENS
        print(n_summary_bits)
        # loop to summarize a clusters summary using looping
        if n_summary_bits > 0:
            for i in range(n_summary_bits):
                bits_summaries = []
                bits_summaries.append(query({"inputs": summary[:MAX_TOKENS]}))
                summary = summary[MAX_TOKENS:]
                print(len(summary))
            bits_summaries.append(query({"inputs": summary})) # For the remaining tokens whose len < MAX_TOKENS
            print(bits_summaries)
            summary = "\n".join(bits[0]['summary_text'] for bits in bits_summaries if isinstance(bits, list)) # Ignore any node which was not summarized because of an API error

        summary = query({"inputs": summary})[0]['summary_text']
        summaries.append(summary)
    print(summaries)
    return summaries

def raptor_template():
    def raptor(doc_splits, questions, embedder, reduced_dim=10, threshold=0.1):
        n_neighbors = int((len(doc_splits) - 1) ** 0.5)
        print(n_neighbors, reduced_dim)
        umap_1 = umap(doc_splits, embedder, n_neighbors, reduced_dim)
        n_clusters = get_optimal_clusters(umap_1, reduced_dim=10)
        responsibilities, _, _, _, _ = gmm(umap_1, n_clusters)
        # Implement soft clusterization by using a threshold
        labels = [np.where(prob > threshold)[0] for prob in responsibilities]
        # Create a mapping of cluster to data points in it
        data_point_to_clusters = {i: label.tolist() for i, label in enumerate(labels)}
        # Get summaries of all the clusters
        cluster_summaries = get_summaries(doc_splits, data_point_to_clusters)

        return cluster_summaries
    return raptor