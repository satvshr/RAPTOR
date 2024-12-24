from utils.gmm import gmm, get_optimal_clusters
from utils.umap import umap
from .raptor import get_summaries
from utils.pdf_summarizer import query
from utils.find_documents import find_documents
from .indexing import extract_questions, get_unique_splits
import numpy as np

def retrieval_template():
    def get_best_nodes(best_cluster_nodes, questions, embedder, top_k, reduced_dim=10, threshold=0.1):
        print("Making the bottom-up tree of nodes belonging to the chosen cluster...")
        questions = extract_questions(questions)
        counter = 1
        # list to summarize nodes of each level only
        current_lvl = best_cluster_nodes.copy()

        # Build a bottom-up tree and use the collapsed tree approach to ensure best performance
        n_clusters = 0
        while True:
            # Print which layer is being processed at the moment
            print(f"Currently layer {counter} is being formed...")
            counter += 1

            # A lot of the code is the same as raptor.py as we are using the same process
            n_neighbors = int((len(current_lvl) - 1) ** 0.5)

            # Perform dimensionality reduction
            lower_dim = umap(current_lvl, embedder, n_neighbors, reduced_dim)

            # Get the number of clusters which would give the best performance
            n_clusters = get_optimal_clusters(lower_dim, reduced_dim)
            if n_clusters == 1: # Break the loop if the optimal cluster is 1
                counter += 1
                print(f"The root node is being formed now, the tree was {counter} layers deep.\n")
                break

            responsibilities, _, _, _, _ = gmm(lower_dim, n_clusters)

            # Implement soft clustering by using a threshold
            labels = [np.where(prob > threshold)[0] for prob in responsibilities]

            # Create a mapping of cluster to data points in it
            data_point_to_clusters = {i: label.tolist() for i, label in enumerate(labels)}

            # Get summaries of all the clusters
            cluster_summaries = get_summaries(current_lvl, data_point_to_clusters, isLayer=True)

            # Update variable to contain the summaries of only the current level
            current_lvl = cluster_summaries

            # Accumalate all nodes of the entire tree under one level (collapsed tree approach) and append it to a list
            best_cluster_nodes.extend(current_lvl)

        # Once the number of optimal clusters is 1, get summaries of the nodes in it and append it to the list
        # This is the root node
        summary = "\n".join(node for node in current_lvl)
        summary = query({"inputs": summary})[0]['summary_text']
        best_cluster_nodes.append(summary)

        print("The tree was formed and flattened as we are using the collapsed tree approach.")
        print("Total number of nodes in the tree: ", len(best_cluster_nodes))
        # Get the unique nodes arranged in descending order of cosine similarity
        nodes = find_documents(best_cluster_nodes, questions, embedder)
        # Return top_k nodes based on preference
        unique_nodes = get_unique_splits(nodes)[:top_k]
        print(f"Selected the {top_k} nodes which could best answer our question using cosine similarity search.\n\n")

        # Return top_k nodes as a string seperated by the new line character 
        return "\n".join(node for node in unique_nodes)

    return get_best_nodes