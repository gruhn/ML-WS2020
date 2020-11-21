import numpy as np
import math

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]
    D = 1
    N = samples.size

    def p(x):
        distances = sorted([ abs(x - xn) for xn in samples ])
        radius = distances[k-1:k][0]
        V = math.pi * radius**2

        return k/(N*V)

    pos = np.arange(-5.0, 5, 0.1)
    est = np.array([ p(x) for x in pos ])
    
    return np.column_stack([ pos, est ])