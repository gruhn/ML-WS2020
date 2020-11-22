import numpy as np

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood
    (N, D) = X.shape

    def multi_gauss(mean, covariance, x): 
        return 1/((2*np.pi)**(D/2) * np.linalg.det(covariance)) * np.exp(-1/2 * np.transpose(x - mean) * np.linalg.inv(covariance) * (x - mean))

    def mixture_density(x):
        return np.sum([ multi_gauss(means[j], covariances[:,:,j], x)*weights[j] for j in range(len(weights)) ])

    logLikelihood = -np.sum([ np.log(mixture_density(x)) for x in X ])

    return logLikelihood

