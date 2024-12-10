import numpy as np
import random

def gmm(documents, n_classes):
    # Dimensionality of points
    dim = documents.shape[1]
    size = documents.shape[0]
    print("doc: ", documents.shape)
    # All pi's should add up to 1
    random_numbers = np.random.rand(n_classes)
    pi = random_numbers / np.sum(random_numbers)

    # Initialize means by randomly choosing data points
    mean = random.sample(documents.tolist(), k=n_classes)  
    print("mean: ", np.array(mean).shape)

    # Initialize covariance matrices by assigning each one as an identity matrix of size (classes, dim, dim)
    cov  = np.tile(np.identity(n_classes), (n_classes, 1, 1))
    print("cov: ", cov.shape)
    # Compute responsibilities
    def expectation(documents, n_classes, pi, mean, cov):
        dif = np.empty([n_classes, size])
        mahalanobis = np.empty([n_classes, size])

        for i in range(n_classes):
            for j in range(size): # To indicate each data point
                dif[i, j] = documents[j] - mean[i]
                print("dif: ", dif.shape)
                mahalanobis[i, j] = np.transpose(dif[i, j]) * np.linalg.inv(cov[i]) * dif[i, j]
                normalization_constant = 1 / (((2 * np.pi) ** dim/2) * np.sqrt(np.linalg.det(cov)))
                exp = np.exp(-1/2 * mahalanobis)

                N = normalization_constant * exp
                gaussian = pi * N
                gaussian_sum = np.sum(gaussian)

                responsibility = gaussian / gaussian_sum
        print(responsibility)
        return responsibility
    expectation(documents, n_classes, pi, mean, cov)

gmm(np.array([[1, 2],[2, 3],[3, 4],[4, 6]]), 2)