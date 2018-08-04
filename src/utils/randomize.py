#
# Import Modules
import numpy as np
#
class Randomizer:
    """
    Randomizer class drawing a random sample from a distribution
    https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
    """
    #
    @staticmethod
    def get_poisson_distribution(p_lam, p_size=None):
        """
        Draw samples from a Poisson distribution.
        """
        return np.random.poisson(lam=p_lam,
                                 size=p_size)
    #
    @staticmethod
    def get_gaussian_distribution(p_loc, p_scale, p_size):
        """
        Draw random samples from a normal (Gaussian) distribution.
        """
        return np.random.normal(loc=p_loc,
                                scale=p_scale,
                                size=p_size)
    #
    @staticmethod
    def get_binomial_distribution(n, p, size=None):
        """
        Draw samples from a binomial distribution.
        """
        return np.random.binomial(n=n,p=p,size=size)