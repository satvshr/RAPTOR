import numpy as np
import random
from langchain_community.embeddings import GPT4AllEmbeddings

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
        
        dif = np.empty([n_classes, size, dim])
        mahalanobis = np.empty([n_classes, size])
        exp = np.empty_like(mahalanobis)
        N = np.empty_like(exp)
        gaussian = np.empty_like(N)

        for i in range(n_classes):
            determinant = np.linalg.det(cov[i])
            determinant = max(determinant, 1e-10)  # Stabilize determinant
            normalization_constant = 1 / (((2 * np.pi) ** (dim/2)) * np.sqrt(determinant))

            for j in range(size): # To indicate each data point
                dif[i, j] = np.array(documents[j] - mean[i])
                mahalanobis[i, j] = np.dot(np.dot(dif[i, j].T, np.linalg.inv(cov[i])), dif[i, j])
                mahalanobis[i, j] = np.clip(mahalanobis[i, j], -1e6, 1e6) # To prevent it from "exploding"

                exp[i, j] = np.exp(-1/2 * mahalanobis[i, j])
                exp[i, j] = np.clip(exp[i, j], 1e-10, None)  # Prevent underflow

                N[i, j] = normalization_constant * exp[i, j]
                gaussian[i, j] = pi[i] * N[i, j]

        gaussian_sums = get_gaussian_sum(gaussian)
        gaussian_sums = np.clip(gaussian_sums, 1e-10, None)
        print("gaussian_sums", gaussian_sums)

        for i in range(n_classes):
            for j in range(size):
                responsibilities[i, j] = gaussian[i, j] / gaussian_sums[j]
        print("mahalanobis", mahalanobis)
        return responsibilities, N, dif

    # Compute pi, mean, and cov using responsibilities obtained
    def maximization(responsibilities, dif):
        new_pi = np.empty_like(pi, dtype=float)
        new_mean = np.empty_like(mean, dtype=float)
        new_cov = np.empty_like(cov, dtype=float)

        for i in range(n_classes):
            resp_sum = np.sum(responsibilities[i])
            new_pi[i] = resp_sum / size
            new_mean[i] = np.sum([responsibilities[i, j] * documents[j] for j in range(size)], axis=0) / resp_sum
            new_cov[i] = np.sum([responsibilities[i, j] * np.outer(dif[i, j], dif[i, j].T) for j in range(size)], axis=0) / resp_sum
            new_cov[i] += np.eye(dim) * 1e-6  # Regularize covariance matrix

        return new_pi, new_mean, new_cov

    def get_log_likelihood(N, pi):
        log_likelihood = 0
        for i in range(size):
            likelihood = 0
            for j in range(n_classes):
                likelihood += pi[j] * N[j, i]
            likelihood = max(likelihood, 1e-10)  # Prevent log(0)
            log_likelihood += np.log(likelihood)

        return log_likelihood

    def EMAlgorithm(responsibilities, pi, mean, cov, max_itt=5, min_gain=1e-6):
        log_likelihood = 0
        _ = 0
        for _ in range(max_itt):
            responsibilities, N, dif = expectation(pi, mean, cov)
            print(responsibilities, N, dif)
            print(111111)
            log_likelihood = get_log_likelihood(N, pi)
            print("log_likelihood", log_likelihood)
            print(2222222)
            pi, mean, cov = maximization(responsibilities, dif)
            print("pi", pi)
            print("mean", mean)
            print("cov", cov)
            print(333333)
            _, N, _ = expectation(pi, mean, cov)    
            gain = get_log_likelihood(N, pi) - log_likelihood
            print("gain", gain)
            print(444444)      
            # if gain < min_gain:
            #     break

        return pi, mean, cov, _
    
    pi, mean, cov, _ = EMAlgorithm(responsibilities, pi, mean, cov)
    print("Final pi", pi)
    print("Final mean", mean)
    print("Final cov", cov)
    print(_)
# Test the function with embeddings
a = np.array(GPT4AllEmbeddings().embed_documents(["a", "b", "d", "e", "f"]))
gmm(a, 3)
