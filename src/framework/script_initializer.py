"""
--------------------------
SCRIPT WARM UP
--------------------------
"""
import os
from src.framework.env_var_loader import ev_loader, ConfigParser
from src.framework.logger import Logger
from src.framework.db_interface import DatabaseInterface
from src.framework.spark import Spark
#
class ScriptInitializer:
    #
    def __init__(self, project_dir, src_dir):
        #
        # Defines config object
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'main/config.ini')
        g_config = ConfigParser(config_file)
        #
        # Loading of program config
        user = str(g_config.get_value('DatabaseConnectionString','user'))
        host = str(g_config.get_value('DatabaseConnectionString','host'))
        service = str(g_config.get_value('DatabaseConnectionString','service'))
        port = str(g_config.get_value('DatabaseConnectionString','port'))
        password = str(g_config.get_value('DatabaseConnectionString','password'))
        instance_name = str(g_config.get_value('DatabaseConnectionString','instance_name'))
        #
        app_name = str(g_config.get_value('SparkContext','app_name'))
        master = str(g_config.get_value('SparkContext','master'))
        spark_rdd_partitions = int(g_config.get_value('SparkContext','spark_rdd_partitions'))
        spark_executor_memory = str(g_config.get_value('SparkContext','spark_executor_memory'))
        spark_executor_cores = str(g_config.get_value('SparkContext','spark_executor_cores'))
        spark_max_result_size = str(g_config.get_value('SparkContext','spark_max_result_size'))
        spark_cores_max = str(g_config.get_value('SparkContext','spark_cores_max'))
        spark_driver_memory = str(g_config.get_value('SparkContext','spark_driver_memory'))
        spark_logConf = str(g_config.get_value('SparkContext','spark_logConf'))
        #
        write_to_disk = str(g_config.get_value("EnvironmentSettings","write_to_disk"))
        write_to_screen = str(g_config.get_value("EnvironmentSettings","write_to_screen"))
        log_file_name = str(g_config.get_value("EnvironmentSettings","log_file_name"))
        #
        tpcds_generation_bool = str(g_config.get_value('DataGeneration','tpcds_data_generation').title())
        tpce_generation_bool = str(g_config.get_value('DataGeneration','tpce_data_generation').title())
        data_generated_directory = str(g_config.get_value('DataGeneration','data_generated_directory'))
        sql_generated_directory = str(g_config.get_value('DataGeneration', 'sql_generated_directory'))
        parallel_degree = int(g_config.get_value('DataGeneration', 'parallel_degree'))
        data_size = int(g_config.get_value('DataGeneration', 'data_size'))
        #
        tpcds_data_loading_bool = str(g_config.get_value('DataLoading','tpcds_loading').title())
        tpce_data_loading_bool = str(g_config.get_value('DataLoading','tpce_loading').title())
        data_generated_dir = str(g_config.get_value('DataGeneration','data_generated_directory'))
        tpcds_sql_generation_bool = str(g_config.get_value('DataGeneration','tpcds_sql_generation').title())
        tpce_sql_generation_bool = str(g_config.get_value('DataGeneration','tpce_sql_generation').title())
        #
        # Load into global dictionary
        ev_loader.var_load({'project_dir':project_dir,
                            'src_dir':src_dir,
                            'user':user,
                            'write_to_disk':write_to_disk,
                            'write_to_screen':write_to_screen,
                            'log_file_name':log_file_name,
                            'instance_name':instance_name,
                            'host':host,
                            'service':service,
                            'port':port,
                            'password':password,
                            'tpcds_generation_bool':tpcds_generation_bool,
                            'tpce_generation_bool':tpce_generation_bool,
                            'data_generated_directory':data_generated_directory,
                            'sql_generated_directory':sql_generated_directory,
                            'parallel_degree':parallel_degree,
                            'data_size':data_size,
                            'tpcds_data_loading_bool':tpcds_data_loading_bool,
                            'tpce_data_loading_bool':tpce_data_loading_bool,
                            'data_generated_dir':data_generated_dir,
                            'tpcds_sql_generation_bool':tpcds_sql_generation_bool,
                            'tpce_sql_generation_bool':tpce_sql_generation_bool,
                            'app_name':app_name.upper(),
                            'master':master,
                            'spark_rdd_partitions':spark_rdd_partitions,
                            'spark_executor_memory':spark_executor_memory,
                            'spark_executor_cores':spark_executor_cores,
                            'spark_max_result_size':spark_max_result_size,
                            'spark_cores_max':spark_cores_max,
                            'spark_driver_memory':spark_driver_memory,
                            'spark_logConf':spark_logConf})
        #
        self.logger = Logger.getInstance(log_file_path=ev_loader.var_get(var_name="project_dir") + "/log/" +
                                          ev_loader.var_get('log_file_name') + "_" + ev_loader.var_get("user") + "_"
                                          + str(Logger.getDate()),
                                         write_to_disk=ev_loader.var_get('write_to_disk'),
                                         write_to_screen=ev_loader.var_get('write_to_screen'))
        #
        self.db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                                         user=ev_loader.var_get('user'),
                                         host=ev_loader.var_get('host'),
                                         service=ev_loader.var_get('service'),
                                         port=ev_loader.var_get('port'),
                                         password=ev_loader.var_get('password'),
                                         logger=self.logger)
        #
        self.spark = Spark(app_name=ev_loader.var_get('app_name'),
                           master=ev_loader.var_get('master'),
                           spark_rdd_partitions=ev_loader.var_get('spark_rdd_partitions'),
                           spark_executor_memory=ev_loader.var_get('spark_executor_memory'),
                           spark_executor_cores=ev_loader.var_get('spark_executor_cores'),
                           spark_max_result_size=ev_loader.var_get('spark_max_result_size'),
                           spark_cores_max=ev_loader.var_get('spark_cores_max'),
                           spark_driver_memory=ev_loader.var_get('spark_driver_memory'),
                           spark_logConf=ev_loader.var_get('spark_logConf'),
                           logger=self.logger)
        self.ev_loader = ev_loader
    #
    def initialize_logger(self):
        """
        Logger Initialization
        :return: logger instance
        """
        return self.logger
    #
    def initialize_database(self):
        """
        Database connection Initialization
        :return:
        """
        return self.db_conn
    #
    def initialize_spark(self):
        """
        Spark Context Initialization
        :return:
        """
        return self.spark
    #
    def get_global_config(self):
        return self.ev_loader