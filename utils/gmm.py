import numpy as np
import random
from langchain_community.embeddings import GPT4AllEmbeddings
from .umap import umap
from scipy.special import logsumexp

def gmm(documents, n_classes):
    # Dimensionality of points
    dim = documents.shape[1]
    size = documents.shape[0]
    print("doc: ", documents.shape)
    # All pi's should add up to 1
    random_numbers = np.random.rand(n_classes)
    pi = np.array(random_numbers / np.sum(random_numbers))
    print("pi", pi.shape)
    # Initialize means by randomly choosing data points
    mean = np.array(random.sample(documents.tolist(), k=n_classes))  
    responsibilities = np.zeros([n_classes, size])

    print("mean: ", mean.shape)

    # Initialize covariance matrices as identity + small regularization
    cov  = np.tile(np.identity(dim), (n_classes, 1, 1))
    cov += np.eye(dim) * 1e-6
    print("cov: ", cov.shape)

    def expectation(pi, mean, cov):
        dif = np.zeros([n_classes, size, dim])
        mahalanobis = np.zeros([n_classes, size])
        log_gaussian = np.zeros([n_classes, size])

        for i in range(n_classes):
            determinant = np.linalg.det(cov[i])
            print("determinant for class", i, ":", determinant)

            half_dim = dim / 2.0
            if determinant <= 0:
                raise ValueError("Determinant not positive. Covariance may not be positive definite.")
            
            log_normalization_constant = - (half_dim * np.log(2 * np.pi) + 0.5 * np.log(determinant))
            print("log_normalization_constant for class", i, ":", log_normalization_constant)

            inv_cov_i = np.linalg.inv(cov[i])

            for j in range(size):
                dif[i, j] = documents[j] - mean[i]
                mahalanobis[i, j] = np.dot(np.dot(dif[i, j].T, inv_cov_i), dif[i, j])
                log_pdf = log_normalization_constant - 0.5 * mahalanobis[i, j]

                # Print some samples to avoid huge logs
                if j < 3:
                    print(f"class {i}, point {j}, log_pdf:", log_pdf)

                log_gaussian[i, j] = np.log(pi[i]) + log_pdf

        print("log_gaussian shape:", log_gaussian.shape)
        print("log_gaussian sample:", log_gaussian[:, :3])

        responsibilities_local = np.zeros([n_classes, size])
        for j in range(size):
            col_logsum = logsumexp(log_gaussian[:, j])
            if j < 3:
                print(f"point {j}, col_logsum:", col_logsum)
            responsibilities_local[:, j] = np.exp(log_gaussian[:, j] - col_logsum)

        N = np.zeros([n_classes, size])
        for i in range(n_classes):
            for j in range(size):
                log_pdf = log_gaussian[i, j] - np.log(pi[i])
                N[i, j] = np.exp(log_pdf)

        return responsibilities_local, N, dif

    def maximization(responsibilities, dif):
        new_pi = np.zeros_like(pi, dtype=float)
        new_mean = np.zeros_like(mean, dtype=float)
        new_cov = np.zeros_like(cov, dtype=float)

        for i in range(n_classes):
            resp_sum = np.sum(responsibilities[i])
            # If cluster is essentially empty, reinitialize it
            if resp_sum < 1e-10:
                new_pi[i] = 1.0 / (size * n_classes)
                new_mean[i] = documents[random.randint(0, size - 1)]
                new_cov[i] = np.eye(dim) * 1e-3
            else:
                new_pi[i] = resp_sum / size
                new_mean[i] = np.sum([responsibilities[i, j] * documents[j] for j in range(size)], axis=0) / resp_sum
                new_cov[i] = np.sum([responsibilities[i, j] * np.outer(dif[i, j], dif[i, j].T) for j in range(size)], axis=0) / resp_sum
                # Increased regularization to maintain positive definiteness
                new_cov[i] += np.eye(dim) * 1e-3

        return new_pi, new_mean, new_cov

    def get_log_likelihood(N, pi):
        log_likelihood = 0
        for i in range(size):
            likelihood = np.sum(pi * N[:, i])
            if likelihood <= 1e-300:
                likelihood = 1e-300
            log_likelihood += np.log(likelihood)
        return log_likelihood

    def EMAlgorithm(responsibilities, pi, mean, cov, max_itt=10, min_gain=1e-6):
        old_log_likelihood = None
        for iteration in range(max_itt):
            print("Iteration:", iteration)
            # E-step
            responsibilities, N, dif = expectation(pi, mean, cov)

            # Compute log-likelihood after E-step
            log_likelihood = get_log_likelihood(N, pi)
            print("log_likelihood", log_likelihood)

            # Check for convergence
            if old_log_likelihood is not None:
                gain = log_likelihood - old_log_likelihood
                print("gain", gain)
                if abs(gain) < min_gain:
                    break

            # M-step
            pi, mean, cov = maximization(responsibilities, dif)
            print("M-pi", pi)
            print("M-mean", mean)
            print("M-cov", cov)

            old_log_likelihood = log_likelihood

        return pi, mean, cov

    pi, mean, cov = EMAlgorithm(responsibilities, pi, mean, cov)
    print("Final pi", pi)
    print("Final mean", mean)
    print("Final cov", cov)

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
a = umap(x, GPT4AllEmbeddings(), 3, 10)
gmm(a, 3)
