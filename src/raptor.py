from utils.umap import umap
from utils.gmm import gmm, get_optimal_clusters
import numpy as np
from sklearn.mixture import GaussianMixture

# @traceable
def raptor_template():
    def raptor(doc_splits, embedder, reduced_dim=10, threshold=0.1):
        n_neighbors = int((len(doc_splits) - 1) ** 0.5)
        print(n_neighbors, reduced_dim)
        umap_1 = umap(doc_splits, embedder, n_neighbors, reduced_dim)
        n_clusters = get_optimal_clusters(umap_1, reduced_dim=10)
        responsibilities, _, _, _, _ = gmm(umap_1, n_clusters)
        labels = [np.where(prob > threshold)[0] for prob in responsibilities]
        # Create a mapping of cluster to data points in it
        data_point_to_clusters = {i: label.tolist() for i, label in enumerate(labels)}
        
        print(data_point_to_clusters)
        return data_point_to_clusters
        
    return raptor