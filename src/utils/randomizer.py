#
# Import Modules
import random
#
class RandomizerUtility:
    #
    @staticmethod
    def randomize(min, max):
        """
        Returns a whole value between two whole numbers
        :param min: Minimum value
        :param max: Maximum value
        :return:
        """
        if min != int(min):
            raise ValueError('Randomised min value must be a whole value!')
        if max != int(max):
            raise ValueError('Randomised max value must be a whole value!')
        #
        return random.randint(min, max)