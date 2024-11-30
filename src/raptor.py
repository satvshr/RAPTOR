from langsmith import Client as traceable
from utils.umap import umap

k_global = 30
k_local = 5
# @traceable
def raptor_template():
    def raptor(doc_splits, embedder, top_k):
        umap_1 = umap(doc_splits, embedder, k_global)
        return doc_splits
        
    return raptor