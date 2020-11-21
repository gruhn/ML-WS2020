import numpy as np
import math


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]
    D = 1
    V = 1
    N = samples.size

    def gaussian_kernel(u):
        return 1/(2 * math.pi * h**2)**(D/2) * math.exp(-u**2/(2 * h**2))

    def p(x):
        K = sum([ gaussian_kernel(x - xn) for xn in samples ])
        return K/(N*V)

    pos = np.arange(-5.0, 5, 0.1)
    est = np.array([ p(x) for x in pos ])
    
    return np.column_stack([ pos, est ])
