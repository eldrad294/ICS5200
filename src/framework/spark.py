#
# Module Imports
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import os
#
class Spark:
    """
    Class dedicated to Spark context manipulation
    http://192.168.202.222:4040/jobs/
    """
    def __init__(self,
                 app_name,
                 master,
                 spark_submit_deployMode,
                 spark_executor_instances,
                 spark_executor_memory,
                 spark_executor_cores,
                 spark_max_result_size,
                 spark_cores_max,
                 spark_driver_memory,
                 spark_default_parallelism,
                 spark_shuffle_partitions,
                 spark_logConf,
                 logger):
        self.__app_name = app_name
        self.__master = master
        self.__spark_submit_deployMode = spark_submit_deployMode
        self.__spark_executor_instances = spark_executor_instances
        self.__spark_executor_memory = spark_executor_memory
        self.__spark_executor_cores = spark_executor_cores
        self.__spark_max_result_size = spark_max_result_size
        self.__spark_cores_max = spark_cores_max
        self.__spark_driver_memory = spark_driver_memory
        self.__spark_default_parallelism = spark_default_parallelism
        self.__spark_shuffle_partitions = spark_shuffle_partitions
        self.__spark_logConf = spark_logConf
        self.__logger = logger
        #
        self.__validate()
        #
        # Initiate master node
        self.__initiate_master_node()
        #
        # Inititate slave node
        self.__initiate_slave_node()
        #
        # Initialize Spark Context
        self.__spark_context = self.__create_Spark_context()
        self.display_spark_context()
        #
        # Initialize Spark Session
        self.__spark_session = self.__create_Spark_session()
    #
    def __validate(self):
        if self.__app_name is None:
            raise ValueError('App Name config was not defined for Spark context!')
        elif self.__master is None:
            raise ValueError('Master config was not declared for Spark context!')
        elif self.__spark_submit_deployMode is None:
            raise ValueError('Spark Deploy Mode was not declared for Spark context!')
        elif self.__spark_executor_instances is None:
            raise ValueError('Spark Executor Instances config was not established!')
        elif self.__spark_driver_memory is None:
            raise ValueError('Spark Executor Memory config was not declared!')
        elif self.__spark_executor_cores is None:
            raise ValueError('Spark Executor Cores config was not declared!')
        elif self.__spark_max_result_size is None:
            raise ValueError('Spark Max Result Set Size config was not declared!')
        elif self.__spark_cores_max is None:
            raise ValueError('Spark Cores Max config was not declared!')
        elif self.__spark_driver_memory is None:
            raise ValueError('Spark Driver Memory config was not declared!')
        elif self.__spark_default_parallelism is None:
            raise ValueError('Spark Default Parallelism was not declared!')
        elif self.__spark_shuffle_partitions is None:
            raise ValueError('Spark Shuffle Partitions was not declared!')
        elif self.__spark_logConf is None:
            raise ValueError('Spark Log Conf config was not declared!')
        elif self.__logger is None:
            raise ValueError("Logger context was not declared!")
    #
    def __create_Spark_context(self):
        """
        Initializes Spark Context
        https://spark.apache.org/docs/latest/configuration.html
        http://alexanderwaldin.github.io/pyspark-quickstart-guide.html
        :return:
        """
        conf = SparkConf()
        conf.setAppName(self.__app_name)
        conf.setMaster(self.__master)
        conf.setExecutorEnv("PYTHONPATH", "$PYTHONPATH:"+os.path.abspath(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir), os.pardir)))
        conf.set('spark.submit.deployMode', str(self.__spark_submit_deployMode))
        conf.set('spark.executor.instances', str(self.__spark_executor_instances))
        conf.set('spark.executor.memory', str(self.__spark_driver_memory))
        conf.set('spark.executor.cores', str(self.__spark_executor_cores))
        conf.set('spark.driver.maxResultSize', str(self.__spark_max_result_size))
        conf.set('spark.cores.max', str(self.__spark_cores_max))
        conf.set('spark.driver.memory', str(self.__spark_driver_memory))
        conf.set('spark.default.parallelism', str(self.__spark_default_parallelism))
        conf.set('spark.sql.shuffle.partitions', str(self.__spark_shuffle_partitions))
        conf.set('spark.logConf', self.__spark_logConf.title())
        sc = SparkContext(conf=conf)
        return sc
    #
    def __create_Spark_session(self):
        return SparkSession(self.__spark_context)
    #
    def get_spark_context(self):
        """
        :return: Instance of Spark Context
        """
        return self.__spark_context
    #
    def get_spark_session(self):
        """
        :return: Instance of Spark Session
        """
        return self.__spark_session
    #
    def display_spark_context(self):
        """
        Displays Spark context
        :return:
        """
        if self.__spark_context is None:
            raise ValueError('Spark Context Uninitialized!')
        #
        for conf in self.__spark_context.getConf().getAll():
            self.__logger.log(conf)
    #
    def __initiate_master_node(self):
        pass
    #
    def __initiate_slave_node(self):
        pass
    #
    def __kill_spark_nodes(self):
        pass
    #
    def __del__(self):
        """
        Before destructing object, ensure that Spark master + slave nodes are killed
        :return:
        """
        self.__kill_spark_nodes()
