from utils.umap import umap
from utils.gmm import gmm, get_optimal_clusters
import numpy as np
from langchain.prompts import PromptTemplate
from utils.lm_studio import LMStudioLLM

def summary_template():
        return PromptTemplate(
        input_variables=["cluster_text"],
        template="""
        System:
        "You are a Summarizing Text Portal."

        User:
        "Write a summary of the following, including as many key details as possible: {cluster_text}"
        """
)

def get_summaries(doc_splits, data_point_to_clusters):
    summaries = []
    lm_studio_llm = LMStudioLLM(path='completions')

    for i in data_point_to_clusters:
        summary = "\n".join(doc_splits[node] for node in data_point_to_clusters[i])
        summary_result = summary_template() | lm_studio_llm
        summary_output = summary_result.invoke(summary)
        summaries.append(summary_output)
    
    return summaries

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
        summaries = get_summaries(doc_splits, data_point_to_clusters)
        print(summaries)

    return raptor