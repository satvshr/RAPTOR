from langsmith import Client as traceable
from utils.gmm import gmm
from utils.umap import umap
k_global = 30
k_local = 5

# @traceable
def raptor_template():
    def raptor(doc_splits, embedder, top_k):
        print(len(doc_splits))
        umap_1 = umap(doc_splits, embedder, 17, 10)
        gmm(umap_1, 3)
        return doc_splits
        
    return raptor