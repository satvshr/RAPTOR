import numpy as np

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

def get_probabilities(neighbours, n_nodes):
    def find_scaling_factor(neighbours_i, target=1.0, tolerance=0.05, step=0.01, max_iterations=1000):
        scaling_factor = 1.0  # Initialization
        iterations = 0
        while True:
            distances = np.array([y for _, y in neighbours_i])
            probabilities_sum = np.sum(np.exp(-distances / scaling_factor))
            if abs(probabilities_sum - target) > tolerance:
                # Adjust scaling_factor based on whether the sum is above or below the target
                scaling_factor = (scaling_factor + step) if (probabilities_sum - target) < 0 else (scaling_factor - step)
                iterations += 1
                if iterations == max_iterations:
                    break
            else:
                break
        return scaling_factor
    
    def symmetrize_probabilities(probabilities):
        symmetric_probabilities = np.zeros_like(probabilities)

        for i in range(n_nodes):
            for j in range(n_nodes):
                symmetric_probabilities[i, j] = probabilities[i, j] + probabilities[j, i] - (probabilities[i, j] * probabilities[j, i])

        return symmetric_probabilities

    # Will be a square symmetric matrix with diagonals and points not falling under the k-point threshold being 0
    probabilities = np.zeros((n_nodes, n_nodes))
    # Find probabilities between points    
    for i in range(n_nodes):
        scaling_factor = find_scaling_factor(neighbours[i])
        for j in range(n_nodes):
            if i != j:  # Same nodes
                distance = next((y for x, y in neighbours[i] if x == j), None)
                if distance is not None:
                    probabilities[i, j] = np.exp(-distance / scaling_factor)
                else:
                    probabilities[i, j] = 0.0
    probabilities = symmetrize_probabilities(probabilities)
    return np.array(probabilities)  

def lower_dim(probabilities, n_nodes, dim, epochs=500, lr=0.1):
    # Matrix of the lower-dim embeddings randomly initialized of size (data points, desired dim)
    low_dim_matrix = np.random.uniform(low=-1, high=1, size=(n_nodes, dim))
    for _ in range(epochs):
        for i in range(n_nodes):
            # grad will be of the same shape as low_dim_matrix
            grad = np.zeros_like(low_dim_matrix[0])
            for j in range(n_nodes):
                if i != j:
                    distance = np.linalg.norm(low_dim_matrix[i] - low_dim_matrix[j])
                    q_ij = 1 / (1 + distance**2) # probabilities of i and j but in the lower dimension
                    grad += 2 * (probabilities[i, j] - q_ij) * ((low_dim_matrix[i] - low_dim_matrix[j])/(1 + distance**2)) # sum up gradient of loss wrt embeddings of point i

            low_dim_matrix[i] += lr * grad
    return low_dim_matrix

def umap(doc_splits, embedder, k, dim):
    # Convert to np array for computation speed
    vectorized_splits = np.array(embedder.embed_documents(doc_splits))
    n_nodes = vectorized_splits.shape[0]
    distances = euclidian_distances(vectorized_splits, n_nodes)
    # Select k-nearest neighbours based on distances
    neighbours = get_neighbours(distances, n_nodes, k)
    # Get the probabilities of i being a meaningful neighbour of j for k-nearest neighbours
    probabilities = get_probabilities(neighbours, n_nodes)
    # Get lower dimension probabilities
    probabilities = lower_dim(probabilities, n_nodes, dim)
    return probabilities