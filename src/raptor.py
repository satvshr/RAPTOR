from langsmith import Client as traceable
from utils.gmm import gmm, get_optimal_clusters
from utils.umap import umap
import numpy as np

# @traceable
def raptor_template():
    def raptor(doc_splits, embedder, reduced_dim=10):
        # Perform soft clustering of the splits
        def soft_clustering(doc_splits, embedder, cluster_doc_mapping, threshold=0.5):
            for i, doc in enumerate(doc_splits):
                for (cluster_idx, mean_embedding), doc_indices in cluster_doc_mapping.items():
                    # Calculate cosine similarity between the document and the mean embedding of the cluster
                    cosine_similarity = np.dot(embedder.embed_query(doc), mean_embedding) / (np.linalg.norm(embedder.embed_query(doc)) * np.linalg.norm(mean_embedding))
                    # If the similarity exceeds the threshold, add the document to the cluster
                    if cosine_similarity > threshold and i not in doc_indices:
                        cluster_doc_mapping[(cluster_idx, mean_embedding)].append(i)

            return cluster_doc_mapping
        
        n_neighbors = int((len(doc_splits) - 1) ** 0.5)
        print(n_neighbors, reduced_dim)
        umap_1 = umap(doc_splits, embedder, n_neighbors, reduced_dim)
        optimal_clusters = get_optimal_clusters(umap_1, reduced_dim)
        responsibilities, _, _, mean, _ = gmm(umap_1, optimal_clusters)
        clusters = np.argmax(responsibilities, axis=0)
        # In the format {(cluster_idx, mean_embedding of the cluster) : [indices of docs part of the cluster], ...}
        cluster_doc_mapping = {(x, tuple(mean[x])):[i for i, j in enumerate(clusters) if j == x] for x in set(clusters)}
        print(cluster_doc_mapping)
        cluster_doc_mapping = soft_clustering(doc_splits, embedder, cluster_doc_mapping)
        print(cluster_doc_mapping)

        return clusters
        
    return raptor