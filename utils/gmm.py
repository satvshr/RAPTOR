import numpy as np
import random

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
            normalization_constant = 1 / (((2 * np.pi) ** (dim/2)) * np.sqrt(np.linalg.det(cov[i])))

            for j in range(size): # To indicate each data point
                dif[i, j] = np.array(documents[j] - mean[i])
                mahalanobis[i, j] = np.dot(np.dot(dif[i, j].T, np.linalg.inv(cov[i])), dif[i, j])
                exp[i, j] = np.exp(-1/2 * mahalanobis[i, j])

                N[i, j] = normalization_constant * exp[i, j]
                gaussian[i, j] = pi[i] * N[i, j]

        gaussian_sums = get_gaussian_sum(gaussian)
        for i in range(n_classes):
            for j in range(size):
                responsibilities[i, j] = gaussian[i, j] / gaussian_sums[j]

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

        return new_pi, new_mean, new_cov

    def get_log_likelihood(N, pi):
        log_likelihood = 0
        for i in range(size):
            likelihood = 0
            for j in range(n_classes):
                likelihood += pi[j] * N[j, i]
            log_likelihood += np.log(likelihood)

        return log_likelihood

    def EMAlgorithm(responsibilities, pi, mean, cov, max_itt=1000, min_gain=1e-6):
        log_likelihood = 0
        for _ in range(max_itt):
            responsibilities, N, dif = expectation(pi, mean, cov)
            log_likelihood = get_log_likelihood(N, pi)
            pi, mean, cov = maximization(responsibilities, dif)
            _, N, _ = expectation(pi, mean, cov)          
            gain = get_log_likelihood(N, pi) - log_likelihood
            if gain < min_gain:
                break

        return pi, mean, cov
    
    pi, mean, cov = EMAlgorithm(responsibilities, pi, mean, cov)
    print(pi, mean, cov)
gmm(np.array([[1, 7],[6, 4],[9, 14],[11, 12]]), 3)