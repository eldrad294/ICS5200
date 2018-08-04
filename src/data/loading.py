from pyspark import SparkContext, SparkConf
from src.framework.logger import logger
from src.framework.config_parser import g_config
from src.utils.db_interface import db_conn
from src.data.spark_maps import SparkMaps
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
        self.__rdd_parallelism = g_config.get_value('SparkContext','rdd_partitions')
        self.__app_name = app_name
        self.__master = master
        self.__validate()
        #
        self.sc = self.__create_Spark_context()
    #
    def __create_Spark_context(self):
        conf = SparkConf()
        conf.setAppName(self.__app_name)
        conf.setMaster(self.__master)
        conf.set('spark.executor.memory', str(g_config.get_value('SparkContext','spark_executor_memory')))
        conf.set('spark.executor.cores', str(g_config.get_value('SparkContext','spark_executor_cores')))
        conf.set('spark.driver.maxResultSize', str(g_config.get_value('SparkContext', 'spark_max_result_size')))
        conf.set('spark.cores.max', str(g_config.get_value('SparkContext','spark_cores_max')))
        conf.set('spark.driver.memory', str(g_config.get_value('SparkContext','spark_driver_memory')))
        conf.set('spark.logConf', g_config.get_value('SparkContext','spark_logConf').title())
        sc = SparkContext(conf=conf)
        logger.log("Spark Context Established..")
        for conf in sc.getConf().getAll():
            logger.log(conf)
        return sc
    #
    def __validate(self):
        if self.__app_name is None:
            raise ValueError('App name was not defined for Spark context!')
        elif self.__master is None:
            raise ValueError('Master was not declared for Spark context!')
        elif self.__rdd_parallelism is None:
            raise ValueError('RDD Parallelism degree was not established!')
        #
        try:
            self.__rdd_parallelism = int(self.__rdd_parallelism)
        except ValueError:
            raise ValueError('RDD Parallelism degree must be a numeric value')
    #
    def load_data(self, path, table_name):
        rdd_file = self.sc.textFile(path, self.__rdd_parallelism) # Materializes an RDD, but does not compute due to lazy evaluation
        rdd_file = rdd_file.map(lambda x: x.split('\n')) # Split line by line - does not compute immediately due to lazy evaluation
        rdd_file.foreachPartition(SparkMaps.build_insert(table_name=table_name))
        db_conn.commit()
        #
        logger.log("Loaded table [" + table_name + "] into database..")

