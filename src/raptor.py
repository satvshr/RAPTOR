from langsmith import Client as traceable
from utils.gmm import gmm, get_optimal_clusters
from utils.umap import umap
import numpy as np

k_global = 30
k_local = 5

# @traceable
def raptor_template():
    def raptor(doc_splits, embedder, reduced_dim=10):
        print(len(doc_splits))
        n_neighbors = (len(doc_splits) - 1) ** 0.5
        umap_1 = umap(doc_splits, embedder, n_neighbors, reduced_dim)
        optimal_clusters = get_optimal_clusters(umap_1, reduced_dim)
        responsibilities, _, _, _, _ = gmm(umap_1, optimal_clusters)
        clusters = np.argmax(responsibilities, axis=0)
        return doc_splits
        
    return raptor