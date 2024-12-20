import numpy as np
import random
from langchain_community.embeddings import GPT4AllEmbeddings
from .umap import umap

def gmm(documents, n_clusters):
    # Dimensionality of points
    dim = documents.shape[1]
    size = documents.shape[0]

    # All pi's should add up to 1
    random_numbers = np.random.rand(n_clusters)
    pi = np.array(random_numbers / np.sum(random_numbers))

    # Initialize means by randomly choosing data points
    mean = np.array(random.sample(documents.tolist(), k=n_clusters))  

    # Initialize responsibilities, which are the probabilities that a data point belongs to a specific cluster
    responsibilities = np.zeros([n_clusters, size])

    # Initialize covariance matrices by assigning each one as an identity matrix of size (classes, dim, dim)
    cov  = np.tile(np.identity(dim), (n_clusters, 1, 1))
    cov += np.eye(dim) * 1e-6

    # Compute responsibilities
    def expectation(pi, mean, cov):
        def get_gaussian_sum(gaussian):
            gaussian_sums = np.zeros([size]) # Sum of gaussian for (all)point j across all i classes
            for i in range(size):
                for j in range(n_clusters):
                    gaussian_sums[i] += gaussian[j, i]

            return gaussian_sums
        
        dif = np.zeros([n_clusters, size, dim])
        mahalanobis = np.zeros([n_clusters, size])
        exp = np.zeros_like(mahalanobis)
        N = np.zeros_like(exp)
        gaussian = np.zeros_like(N)

        for i in range(n_clusters):
            determinant = np.linalg.det(cov[i])
            normalization_constant = 1 / (((2 * np.pi) ** (dim/2)) * np.sqrt(determinant))

            for j in range(size): # each data point
                dif[i, j] = np.array(documents[j] - mean[i])
                mahalanobis[i, j] = np.dot(np.dot(dif[i, j].T, np.linalg.inv(cov[i])), dif[i, j])

                exp[i, j] = np.exp(-0.5 * mahalanobis[i, j])

                N[i, j] = normalization_constant * exp[i, j]
                gaussian[i, j] = pi[i] * N[i, j]

        gaussian_sums = get_gaussian_sum(gaussian)

        for i in range(n_clusters):
            for j in range(size):
                responsibilities[i, j] = gaussian[i, j] / (gaussian_sums[j] + 1e-12) # Prevent division by 0
        return responsibilities, N, dif

    # Compute pi, mean, and cov using responsibilities obtained
    def maximization(responsibilities, dif):
        new_pi = np.zeros_like(pi, dtype=float)
        new_mean = np.zeros_like(mean, dtype=float)
        new_cov = np.zeros_like(cov, dtype=float)

        for i in range(n_clusters):
            resp_sum = np.sum(responsibilities[i])
            new_pi[i] = resp_sum / size
            new_mean[i] = np.sum([responsibilities[i, j] * documents[j] for j in range(size)], axis=0) / resp_sum
            new_cov[i] = np.sum([responsibilities[i, j] * np.outer(dif[i, j], dif[i, j].T) for j in range(size)], axis=0) / resp_sum
            new_cov[i] += np.eye(dim) * 1e-6  # Regularize covariance

        return new_pi, new_mean, new_cov

    def get_log_likelihood(N, pi):
        log_likelihood = 0
        for i in range(size):
            likelihood = 0
            for j in range(n_clusters):
                likelihood += pi[j] * N[j, i]
                likelihood = max(likelihood, 1e-12) # To ensure it is not zero

            log_likelihood += np.log(likelihood)

        return log_likelihood

    def EMAlgorithm(pi, mean, cov, max_itt=1000, min_gain=1e-6):
        old_log_likelihood = None
        for _ in range(max_itt):
            # E-step
            responsibilities, N, dif = expectation(pi, mean, cov)

            # Compute log-likelihood after E-step
            log_likelihood = get_log_likelihood(N, pi)

            # Check for convergence
            if old_log_likelihood is not None:
                gain = log_likelihood - old_log_likelihood
                if gain < min_gain:
                    break

            # M-step
            pi, mean, cov = maximization(responsibilities, dif)
            old_log_likelihood = log_likelihood

        return responsibilities, log_likelihood, pi, mean, cov

    responsibilities, log_likelihood, pi, mean, cov = EMAlgorithm(pi, mean, cov)

    return responsibilities, log_likelihood, pi, mean, cov

# Get optimal number of clusters using BIC
def get_optimal_clusters(embeddings, reduced_dim, max_clusters=10):
    bic_scores = []
    for n_clusters in range(1, max_clusters+1):
        _, log_likelihood, _, _, _ = gmm(embeddings, n_clusters)
        mean_params = n_clusters * reduced_dim
        cov_params = n_clusters * (reduced_dim * (reduced_dim + 1) // 2)
        pi_params = n_clusters - 1
        k = mean_params + cov_params + pi_params # Total number of params
        bic = -2 * log_likelihood + k * np.log(len(embeddings))
        bic_scores.append(bic)

    return np.argmin(bic_scores) + 1