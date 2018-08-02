#
# Import Modules
import numpy as np
#
class Randomizer:
    """
    Randomizer class drawing a random sample from a distribution
    """
    #
    @staticmethod
    def get_poisson_distribution(p_lam, p_size=None):
        """
        Draw samples from a Poisson distribution.

        :param p_lam:  float or array_like of floats
                       Expectation of interval, should be >= 0. A sequence of expectation intervals must be
                       broadcastable over the requested size.
        :param p_size:  int or tuple of ints, optional Output shape.
                        If the given shape is, e.g., (m, n, k), then
                        m * n * k samples are drawn. If size is None (default), a single value is returned if lam is a scalar.
                        Otherwise, np.array(lam).size samples are drawn
        :return: ndarray or scalar
                 Drawn samples from the parameterized Poisson distribution
        """
        return np.random.poisson(lam=p_lam,
                                 size=p_size)
    #
    @staticmethod
    def get_gaussian_distribution(p_loc, p_scale, p_size):
        """
        Draw random samples from a normal (Gaussian) distribution.

        :param p_loc: float or array_like of floats
                      Mean (“centre”) of the distribution.
        :param p_scale: float or array_like of floats
                        Standard deviation (spread or “width”) of the distribution.
        :param p_size: int or tuple of ints, optional
                       Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.
        :return: ndarray or scalar
                 Drawn samples from the parameterized normal distribution.
        """
        return np.random.normal(loc=p_loc,
                                scale=p_scale,
                                size=p_size)
    #
    @staticmethod
    def get_binomial_distribution(n, p, size=None):
        """
        Draw samples from a binomial distribution.
        http://www.stat.yale.edu/Courses/1997-98/101/binom.htm

        :param n: Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.
        :param p: Parameter of the distribution, >= 0 and <=1.
        :param size: Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is
                     None (default), a single value is returned if n and p are both scalars. Otherwise,
                     np.broadcast(n, p).size samples are drawn.
        :return:
        """
        return np.random.binomial(n=n,p=p,size=size)