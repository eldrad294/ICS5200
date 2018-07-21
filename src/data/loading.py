from pyspark import SparkContext, SparkConf
from src.framework.logger import logger
#
# Module Imports
class FileLoader:
    """
    This class is reserved for the un/loading of data into the database instance. The class is particularly customized
    for TPC generated data, allowing .dat files to be parsed, and loaded into database tables. It utilizes the Spark
    toolset to manipulate file un/loading in an efficient manner.
    """
    #
    def __init__(self, app_name="ICS5200", master="local"):
        #
        self.__validate(app_name=app_name,
                        master=master)
        #
        conf = SparkConf().setAppName(app_name).setMaster(master)
        self.sc = SparkContext(conf=conf)
        logger.log("Spark Context Established..")
    #
    def __validate(self, app_name, master):
        if app_name is None:
            raise Exception('App name was not defined for Spark context!')
        elif master is None:
            raise Exception('Master was not declared for Spark context!')
    #
    def load_data(self, path):
        dist_file = self.sc.textFile(path)
        print(dist_file)
