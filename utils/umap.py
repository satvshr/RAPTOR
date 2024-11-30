import numpy as np
from langchain_community.embeddings import GPT4AllEmbeddings

def euclidian_distances(vectors, n_nodes):
    # Will be a square symmetric matrix with diagonals being 0
    distances = np.zeros((n_nodes, n_nodes))
    # Find distance between points
    for i in range(n_nodes):
        # No need to recalculate already calculated distances so skip points before the i-th one
        for j in range(i+1, n_nodes):
            distances[i, j] = np.linalg.norm((vectors[i] - vectors[j]) ** 2)
            distances[j, i] = distances[i, j]

    return distances

def get_neighbours(distances, n_nodes, k):
    # Store a dictionary having node as key and [(point, distance), ...] as values
    neighbours = {}
    # Get k sorted (index, element) pair for all nodes 
    for i in range(n_nodes):
        neighbours[i] = sorted(enumerate(distances[i]), key=lambda x:x[1])[:k]

    return neighbours

def get_probablities(neighbours, n_nodes):
    # Will be a square symmetric matrix with diagonals being 0
    probablities = np.zeros((n_nodes, n_nodes))
    # Find probablities between points    
    for i in range(n_nodes):
        for j in range(n_nodes):
            while i != j: # Same nodes
                probablities[i, j] = 
    return probablities
    
def umap(doc_splits, embedder, k):
    # Convert to np array for computation speed
    vectorized_splits = np.array(embedder.embed_documents(doc_splits))
    n_nodes = vectorized_splits.shape[0]
    distances = euclidian_distances(vectorized_splits, n_nodes)
    # Select k-nearest neighbours based on distances
    neighbours = get_neighbours(distances, n_nodes, k)
    # Get the probablities of i being a meaningful enighbour of j for k-nearest neighbours
    probablities = get_probablities(neighbours, n_nodes)
umap(['a', 'b', 'c'], GPT4AllEmbeddings(), 2)