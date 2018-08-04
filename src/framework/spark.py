#
# Module Imports
from src.framework.logger import logger
#
class Spark:
    """
    Class dedicated to Spark context manipulation
    """
    def __init__(self,
                 app_name,
                 master,
                 spark_rdd_partitions,
                 spark_executor_memory,
                 spark_executor_cores,
                 spark_max_result_size,
                 spark_cores_max,
                 spark_driver_memory,
                 spark_logConf,
                 logger):
        self.__app_name = app_name
        self.__master = master
        self.__spark_rdd_partitions = spark_rdd_partitions
        self.__spark_executor_memory = spark_executor_memory
        self.__spark_executor_cores = spark_executor_cores
        self.__spark_max_result_size = spark_max_result_size
        self.__spark_cores_max = spark_cores_max
        self.__spark_driver_memory = spark_driver_memory
        self.__spark_logConf = spark_logConf
        self.__logger = logger
        #
        self.__validate()
        #
        # Initialize Spark Context
        self.__spark_context = self.__create_Spark_context()
    #
    def __validate(self):
        if self.__app_name is None:
            raise ValueError('App Name config was not defined for Spark context!')
        elif self.__master is None:
            raise ValueError('Master config was not declared for Spark context!')
        elif self.__spark_rdd_partitions is None:
            raise ValueError('RDD Spark RDD partition config was not established!')
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
        elif self.__spark_logConf is None:
            raise ValueError('Spark Log Conf config was not declared!')
        elif self.__logger is None:
            raise ValueError("Logger context was not declared!")
        #
        try:
            self.__spark_rdd_partitions = int(self.__spark_rdd_partitions)
        except ValueError:
            raise ValueError('RDD Parallelism degree must be a numeric value')
    #
    def __create_Spark_context(self):
        """
        Initializes Spark Context
        :return:
        """
        conf = SparkConf()
        conf.setAppName(self.__app_name)
        conf.setMaster(self.__master)
        conf.set('spark.executor.memory', str(self.__spark_driver_memory))
        conf.set('spark.executor.cores', str(self.__spark_executor_cores))
        conf.set('spark.driver.maxResultSize', str(self.__spark_max_result_size))
        conf.set('spark.cores.max', str(self.__spark_cores_max))
        conf.set('spark.driver.memory', str(self.__spark_driver_memory))
        conf.set('spark.logConf', self.__spark_logConf.title())
        sc = SparkContext(conf=conf)
        self.display_spark_context()
        return sc
    #
    def get_spark_context(self):
        """
        Returns instance of Spark Context
        :return:
        """
        return self.__spark_context
    #
    def display_spark_context(self):
        """
        Displays Spark context
        :return:
        """
        if self.__spark_context is None:
            raise ValueError('Spart Context Uninitialized!')
        #
        for conf in sc.getConf().getAll():
            self.__logger.log(conf)
#
class SparkMaps:
    """
    This class contains all mapping functions which are utilized throughout this project.
    """
    #
    @staticmethod
    def build_insert(dataline, table_name, database_context, ):
        """
        Formats insert statement
        :param line: Current .DAT line
        :param table: Table data being loaded into
        :param db_conn: Database connection context
        :param spark_context: Spark connection context
        :return:
        """
        l_line = SparkMaps.__parse_data_line(dataline)
        dml = "INSERT INTO " + table_name + " VALUES ("
        for i in range(len(l_line)):
            if i == 0:
                dml += " :" + str(i+1) + " "
            else:
                dml += ", :" + str(i+1) + " "
        dml += ")"
        database_context.execute_dml(dml, l_line).commit()
    #
    @staticmethod
    def __parse_data_line(line):
        """
        Iterates over input data line, and parses value into a list. Values are delimeted according to config file,
        default to '|'
        :param line:
        :return:
        """
        list_line = []
        delimeter = '|'
        value = ""
        for i in line:
            if i != delimeter:
                value += i
            else:
                try:
                    value = int(value)
                except Exception:
                    try:
                        value = float(value)
                    except Exception:
                        pass
                #
                list_line.append(value)
                value = ""
        return tuple(list_line)
