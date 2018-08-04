from pyspark import SparkContext, SparkConf
from src.framework.logger import logger
from src.framework.config_parser import g_config
from src.utils.db_interface import db_conn
from src.data.pyspark_loading import FileLoaderUtils
#
# Module Imports
class FileLoader:
    """
    This class is reserved for the un/loading of data into the database instance. The class is particularly customized
    for TPC generated data, allowing .dat files to be parsed, and loaded into database tables. It utilizes the Spark
    toolset to manipulate file un/loading in an efficient manner.
    """
    #
    def __init__(self, app_name="ICS5200", master="local", db_conn=None):
        #
        self.__validate(app_name=app_name, master=master, db_conn=db_conn)
        self.__db_conn = db_conn
        #
        self.sc = self.__create_Spark_context(app_name=app_name, master=master)
    #
    def __create_Spark_context(self, app_name, master):
        conf = SparkConf()
        conf.setAppName(app_name)
        conf.setMaster(master)
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
    def __validate(self, app_name, master, db_conn):
        if app_name is None:
            raise Exception('App name was not defined for Spark context!')
        elif master is None:
            raise Exception('Master was not declared for Spark context!')
        #
        if db_conn is None:
            raise Exception('Uninitialized database connection!')
    #
    def load_data(self, path, table_name):
        rdd_file = self.sc.textFile(path,1) # Materializes an RDD, but does not compute due to lazy evaluation
        # l_dist_file = dist_file.collect() # Convert into python collection (list)
        # logger.log("Loaded [" + path + "] into memory..")
        # for i, line in enumerate(l_dist_file):
        #     dml, bind_values = self.__build_insert(line, table_name)
        #     db_conn.execute_dml(dml, bind_values)
        #     if i % 10000 == 0 and i != 0:
        #         logger.log("Loaded " + str(i) + " records..")
        # db_conn.commit()
        #
        rdd_file.map(lambda x: x.split('\n')) # Split line by line - does not compute immediately due to lazy evaluation
        rdd_file.foreach(FileLoaderUtils.build_insert)
        self.__db_conn.commit()
        #
        logger.log("Loaded table [" + table_name + "] into database..")

