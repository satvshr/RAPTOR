from langsmith import Client as traceable
from utils.gmm import gmm

k_global = 30
k_local = 5

# @traceable
def raptor_template():
    def raptor(doc_splits, embedder, top_k):
        print(len(doc_splits))
        # umap_1 = gmm(doc_splits, 
        return doc_splits
        
    return raptor