from pyspark import SparkContext, SparkConf
from src.framework.logger import logger
from src.framework.config_parser import g_config
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
        self.__validate(app_name=app_name, master=master)
        #
        self.sc = self.__create_Spark_context(app_name=app_name,master=master)
        #
        self.__bulk_load = 100
        self.__delimeter = '|'
    #
    def __create_Spark_context(self, app_name, master):
        conf = SparkConf()
        conf.setAppName(app_name)
        conf.setMaster(master)
        conf.set('spark.executor.memory', str(g_config.get_value('SparkContext','spark_executor_memory')))
        conf.set('spark.executor.cores', str(g_config.get_value('SparkContext','spark_executor_cores')))
        conf.set('spark.cores.max', str(g_config.get_value('SparkContext','spark_cores_max')))
        conf.set('spark.driver.memory', str(g_config.get_value('SparkContext','spark_driver_memory')))
        conf.set('spark.logConf', g_config.get_value('SparkContext','spark_logConf').title())
        sc = SparkContext(conf=conf)
        logger.log("Spark Context Established..")
        for conf in sc.getConf().getAll():
            logger.log(conf)
        return sc
    #
    def __validate(self, app_name, master):
        if app_name is None:
            raise Exception('App name was not defined for Spark context!')
        elif master is None:
            raise Exception('Master was not declared for Spark context!')
    #
    def load_data(self, path, table_name, db_conn):
        dist_file = self.sc.textFile(path)
        l_dist_file = dist_file.collect() # Convert into python collection (list)
        for i, line in enumerate(l_dist_file):
            line = line.encode('utf-8')
            dml, bind_values = self.__build_insert(line, table_name)
            db_conn.execute_dml(dml, bind_values)
            if i%10000==0 and i != 0:
                logger.log("Loaded " + str(i) + " records..")
        db_conn.commit()
        logger.log("Loaded table [" + table_name + "]")
    #
    def __build_insert(self, line, table):
        """
        Formats insert statement
        :param line:
        :param table:
        :return:
        """
        l_line = self.__parse_data_line(line)
        dml = "INSERT INTO " + table + " VALUES ("
        for i in range(len(l_line)):
            if i == 0:
                dml += " :" + str(i+1) + " "
            else:
                dml += ", :" + str(i+1) + " "
        dml += ")"
        return dml, l_line
    #
    def __parse_data_line(self, line):
        """
        Iterates over input data line, and parses value into a list. Values are delimeted according to config file,
        default to '|'
        :param line:
        :return:
        """
        list_line = []
        value = ""
        for i in line:
            if i != self.__delimeter:
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

