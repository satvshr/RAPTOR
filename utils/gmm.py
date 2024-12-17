import numpy as np
import random
from langchain_community.embeddings import GPT4AllEmbeddings
from .umap import umap

def gmm(documents, n_classes):
    # Dimensionality of points
    dim = documents.shape[1]
    size = documents.shape[0]
    print("doc: ", documents.shape)
    # All pi's should add up to 1
    random_numbers = np.random.rand(n_classes)
    pi = np.array(random_numbers / np.sum(random_numbers))
    print("pi: ", pi.shape)
    # Initialize means by randomly choosing data points
    mean = np.array(random.sample(documents.tolist(), k=n_classes))  
    responsibilities = np.zeros([n_classes, size])

    print("mean: ", mean.shape)

    # Initialize covariance matrices by assigning each one as an identity matrix of size (classes, dim, dim)
    cov  = np.tile(np.identity(dim), (n_classes, 1, 1))
    cov += np.eye(dim) * 1e-6
    print("cov: ", cov.shape)

    # Compute responsibilities
    def expectation(pi, mean, cov):
        def get_gaussian_sum(gaussian):
            gaussian_sums = np.zeros([size]) # Sum of gaussian for (all)point j across all i classes
            for i in range(size):
                for j in range(n_classes):
                    gaussian_sums[i] += gaussian[j, i]
            return gaussian_sums
        
        dif = np.zeros([n_classes, size, dim])
        mahalanobis = np.zeros([n_classes, size])
        exp = np.zeros_like(mahalanobis)
        N = np.zeros_like(exp)
        gaussian = np.zeros_like(N)

        for i in range(n_classes):
            determinant = np.linalg.det(cov[i])
            # determinant = max(determinant, 1e-10)  # Stabilize determinant
            print("determinant\n", determinant)
            normalization_constant = 1 / (((2 * np.pi) ** (dim/2)) * np.sqrt(determinant))

            for j in range(size): # each data point
                dif[i, j] = np.array(documents[j] - mean[i])
                mahalanobis[i, j] = np.dot(np.dot(dif[i, j].T, np.linalg.inv(cov[i])), dif[i, j])
                # mahalanobis[i, j] = np.clip(mahalanobis[i, j], -1e6, 1e6) # Just to avoid extreme values

                exp[i, j] = np.exp(-0.5 * mahalanobis[i, j])
                # exp[i, j] = np.clip(exp[i, j], 1e-10, None)  # Prevent underflow

                N[i, j] = normalization_constant * exp[i, j]
                gaussian[i, j] = pi[i] * N[i, j]

        gaussian_sums = get_gaussian_sum(gaussian)
        print("mahalanobis\n", mahalanobis)
        print("Normalization constant\n", normalization_constant)
        print("exp\n", exp)
        print("N\n", N)
        print("pi\n", pi)
        print("gaussian\n", gaussian)
        print("gaussian_sums\n", gaussian_sums)

        for i in range(n_classes):
            for j in range(size):
                responsibilities[i, j] = gaussian[i, j] / gaussian_sums[j]
        return responsibilities, N, dif

    # Compute pi, mean, and cov using responsibilities obtained
    def maximization(responsibilities, dif):
        new_pi = np.zeros_like(pi, dtype=float)
        new_mean = np.zeros_like(mean, dtype=float)
        new_cov = np.zeros_like(cov, dtype=float)

        for i in range(n_classes):
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
            for j in range(n_classes):
                likelihood += pi[j] * N[j, i]
            # likelihood = max(likelihood, 1e-10)  # Prevent log(0)
            log_likelihood += np.log(likelihood)
        return log_likelihood

    def EMAlgorithm(responsibilities, pi, mean, cov, max_itt=6, min_gain=1e-6):
        old_log_likelihood = None
        for _ in range(max_itt):
            # E-step
            responsibilities, N, dif = expectation(pi, mean, cov)

            # Compute log-likelihood after E-step
            log_likelihood = get_log_likelihood(N, pi)
            print("log_likelihood\n", log_likelihood)

            # Check for convergence
            if old_log_likelihood is not None:
                gain = log_likelihood - old_log_likelihood
                print("gain\n", gain)
                # if gain < min_gain:
                #     break

            # M-step
            pi, mean, cov = maximization(responsibilities, dif)
            # print("M-pi", pi)
            # print("M-mean", mean)
            # print("M-cov", cov)
            old_log_likelihood = log_likelihood

        return pi, mean, cov

    pi, mean, cov = EMAlgorithm(responsibilities, pi, mean, cov)
    print("Final pi\n", pi)
    print("Final mean\n", mean)
    print("Final cov\n", cov)

# Test the function with embeddings
x = [
    "The sun rises over the peaceful mountains.",  # Cluster 1
    "Birds sing in the quiet morning forest.",     # Cluster 1
    "The gentle stream flows through the valley.", # Cluster 1
    "Cars honk loudly on the crowded city streets.",  # Cluster 2
    "The subway station is packed with commuters.",   # Cluster 2
    "Bright billboards light up the bustling downtown.", # Cluster 2
    "People rush to catch the next train in the city."   # Cluster 2
]
a = umap(x, GPT4AllEmbeddings(), 17, 10)
gmm(a, 3)