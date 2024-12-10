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
    mean = random.sample(documents.tolist(), k=n_classes)  # Convert documents to list
    print("mean: ", np.array(mean).shape)

    # Initialize covariance matrices by assigning each one as an identity matrix of size (classes, dim, dim)
    cov  = np.tile(np.identity(n_classes), (n_classes, 1, 1))
    print("cov: ", cov.shape)
    # Compute responsibilities
    def expectation(documents, n_classes, pi, mean, cov):
        dif = np.empty([n_classes, size, dim])
        mahalanobis = np.empty([n_classes, size])
        exp = np.empty_like(mahalanobis)
        N = np.empty_like(exp)
        gaussian = np.empty_like(N)
        responsibility = np.empty_like(gaussian)

        for i in range(n_classes):
            normalization_constant = 1 / (((2 * np.pi) ** (dim/2)) * np.sqrt(np.linalg.det(cov[i])))
            gaussian_sum = 0 # Sum of gaussian for point j across all i classes

            for j in range(size): # To indicate each data point
                dif[i, j] = np.array(documents[j] - mean[i])
                print("dif: ", dif.shape)
                mahalanobis[i, j] = np.dot(np.dot(dif[i, j].T, np.linalg.inv(cov[i])), dif[i, j])
                exp[i, j] = np.exp(-1/2 * mahalanobis[i, j])

                N[i, j] = normalization_constant * exp[i, j]
                gaussian[i, j] = pi[i] * N[i, j]
                gaussian_sum += gaussian[i, j]

                responsibility[i, j] = gaussian[i, j] / gaussian_sum
        print(responsibility)
        return responsibility
    expectation(documents, n_classes, pi, mean, cov)

gmm(np.array([[1, 2],[2, 3],[3, 4],[4, 6]]), 2)