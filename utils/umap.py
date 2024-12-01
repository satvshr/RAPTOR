import numpy as np
from langchain_community.embeddings import GPT4AllEmbeddings

def euclidian_distances(vectors, n_nodes):
    # Will be a square symmetric matrix with diagonals being infinity so that the points never fall under the k-nearest threshold
    distances = np.zeros((n_nodes, n_nodes))
    # Find distance between points
    for i in range(n_nodes):
        # No need to recalculate already calculated distances so skip points before the i-th one
        for j in range(i+1, n_nodes):
            distances[i, j] = np.linalg.norm(vectors[i] - vectors[j])
            distances[j, i] = distances[i, j]
        distances[i, i] = np.inf

    return distances

def get_neighbours(distances, n_nodes, k):
    # Store a dictionary having node as key and [(point, distance), ...] as values
    neighbours = {}
    # Get k sorted (index, element) pair for all nodes 
    for i in range(n_nodes):
        neighbours[i] = sorted(enumerate(distances[i]), key=lambda x:x[1])[:k]

    return neighbours

def get_probablities(neighbours, n_nodes):
    def find_scaling_factor(neighbours_i, target=1.0, tolerance=0.05, step=0.01, max_iterations=1000):
        scaling_factor = 1.0  # Initialization
        iterations = 0
        while True:
            distances = np.array([y for _, y in neighbours_i])
            probablities_sum = np.sum(np.exp(-distances / scaling_factor))
            if abs(probablities_sum - target) > tolerance:
                # Adjust scaling_factor based on whether the sum is above or below the target
                scaling_factor = (scaling_factor + step) if (probablities_sum - target) < 0 else (scaling_factor - step)
                iterations += 1
                if iterations == max_iterations:
                    break
            else:
                break
        return scaling_factor

    # Will be a square symmetric matrix with diagonals and points not falling under the k-point threshold being 0
    probablities = np.zeros((n_nodes, n_nodes))
    # Find probablities between points    
    for i in range(n_nodes):
        scaling_factor = find_scaling_factor(neighbours[i])
        for j in range(n_nodes):
            if i != j:  # Same nodes
                distance = next((y for x, y in neighbours[i] if x == j), None)
                if distance is not None:
                    probablities[i, j] = np.exp(-distance / scaling_factor)
                else:
                    probablities[i, j] = 0.0
    print(probablities)
    return probablities  

def umap(doc_splits, embedder, k):
    # Convert to np array for computation speed
    vectorized_splits = np.array(embedder.embed_documents(doc_splits))
    n_nodes = vectorized_splits.shape[0]
    distances = euclidian_distances(vectorized_splits, n_nodes)
    # Select k-nearest neighbours based on distances
    neighbours = get_neighbours(distances, n_nodes, k)
    # Get the probablities of i being a meaningful enighbour of j for k-nearest neighbours
    get_probablities(neighbours, n_nodes)

umap(['a', 'b', 'c', 'd'], GPT4AllEmbeddings(), 2)