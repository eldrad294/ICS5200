import os
from src.framework.env_var_loader import ev_loader, ConfigParser
from src.framework.logger import Logger
from src.framework.db_interface import DatabaseInterface, ConnectionPool
from src.framework.spark import Spark
#
class ScriptInitializer:
    """
    This class serves as a wrapper to all scripts. The class, when initialized, will establish all the required config
    for the environment to operate. This class should be initialized only once, at the beginning of every script.
    """
    #
    def __init__(self, project_dir, src_dir, home_dir):
        #
        # Defines config object
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'main/config.ini')
        g_config = ConfigParser(config_file)
        #
        # Environment var loading
        os.environ['PYSPARK_PYTHON'] = project_dir + '/venv/bin/python3'
        os.environ['PYSPARK_DRIVER_PYTHON'] = project_dir + '/venv/bin/python3'
        os.environ['SPARK_YARN_USER_ENV'] = project_dir + '/venv/bin/python3'
        #
        # Loading of Oracle config
        oracle_home = str(g_config.get_value('Oracle','oracle_home'))
        ld_library_path = str(g_config.get_value('Oracle','ld_library_path'))
        #
        # Loading of database config
        user = g_config.get_value('DatabaseConnectionString','user')
        host = g_config.get_value('DatabaseConnectionString','host')
        service = g_config.get_value('DatabaseConnectionString','service')
        port = g_config.get_value('DatabaseConnectionString','port')
        password = g_config.get_value('DatabaseConnectionString','password')
        instance_name = g_config.get_value('DatabaseConnectionString','instance_name')
        #
        # Spark Config
        app_name = g_config.get_value('SparkContext','app_name')
        master = g_config.get_value('SparkContext','master')
        spark_installation_path = g_config.get_value('SparkContext','spark_installation_path')
        spark_submit_deployMode = g_config.get_value('SparkContext','spark_submit_deployMode')
        spark_executor_instances = g_config.get_value('SparkContext','spark_executor_instances')
        spark_executor_memory = g_config.get_value('SparkContext','spark_executor_memory')
        spark_executor_cores = g_config.get_value('SparkContext','spark_executor_cores')
        spark_max_result_size = g_config.get_value('SparkContext','spark_max_result_size')
        spark_cores_max = g_config.get_value('SparkContext','spark_cores_max')
        spark_driver_memory = g_config.get_value('SparkContext','spark_driver_memory')
        spark_default_parallelism = g_config.get_value('SparkContext','spark_default_parallelism')
        spark_shuffle_partitions = g_config.get_value('SparkContext','spark_shuffle_partitions')
        spark_logConf = g_config.get_value('SparkContext','spark_logConf')
        spark_python_worker_reuse = g_config.get_value('SparkContext','spark_python_worker_reuse')
        #
        write_to_disk = g_config.get_value("EnvironmentSettings","write_to_disk")
        write_to_screen = g_config.get_value("EnvironmentSettings","write_to_screen")
        log_file_name = g_config.get_value("EnvironmentSettings","log_file_name")
        #
        # Data Generation Config
        data_retain_bool = str(g_config.get_value('DataGeneration','data_retain')).title()
        tpcds_generation_bool = str(g_config.get_value('DataGeneration','tpcds_data_generation').title())
        tpce_generation_bool = str(g_config.get_value('DataGeneration','tpce_data_generation').title())
        data_generated_directory = str(g_config.get_value('DataGeneration','data_generated_directory'))
        sql_generated_directory = str(g_config.get_value('DataGeneration', 'sql_generated_directory'))
        parallel_degree = int(g_config.get_value('DataGeneration', 'parallel_degree'))
        data_size = int(g_config.get_value('DataGeneration', 'data_size'))
        #
        # Data Loading Config
        tpcds_data_loading_bool = str(g_config.get_value('DataLoading','tpcds_loading').title())
        tpce_data_loading_bool = str(g_config.get_value('DataLoading','tpce_loading').title())
        data_generated_dir = str(g_config.get_value('DataGeneration','data_generated_directory'))
        tpcds_sql_generation_bool = str(g_config.get_value('DataGeneration','tpcds_sql_generation').title())
        tpce_sql_generation_bool = str(g_config.get_value('DataGeneration','tpce_sql_generation').title())
        #
        # Load into global dictionary
        ev_loader.var_load({'project_dir':project_dir,
                            'src_dir':src_dir,
                            'home_dir':home_dir,
                            'spark_installation_path':spark_installation_path,
                            'user':user,
                            'write_to_disk':write_to_disk,
                            'write_to_screen':write_to_screen,
                            'log_file_name':log_file_name,
                            'oracle_home':oracle_home,
                            'ld_library_path':ld_library_path,
                            'instance_name':instance_name,
                            'host':host,
                            'service':service,
                            'port':port,
                            'password':password,
                            'data_retain_bool':data_retain_bool,
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
                            'spark_submit_deployMode':spark_submit_deployMode,
                            'spark_executor_instances':spark_executor_instances,
                            'spark_executor_memory':spark_executor_memory,
                            'spark_executor_cores':spark_executor_cores,
                            'spark_max_result_size':spark_max_result_size,
                            'spark_cores_max':spark_cores_max,
                            'spark_driver_memory':spark_driver_memory,
                            'spark_default_parallelism':spark_default_parallelism,
                            'spark_shuffle_partitions':spark_shuffle_partitions,
                            'spark_logConf':spark_logConf,
                            'spark_python_worker_reuse':spark_python_worker_reuse,
                            'log_file_path':project_dir + "/log/" + log_file_name + "_" + user})
        #
        self.logger = Logger(log_file_path=ev_loader.var_get('log_file_path'),
                             write_to_disk=ev_loader.var_get('write_to_disk'),
                             write_to_screen=ev_loader.var_get('write_to_screen'))
        #
        connection_details = {'instance_name':ev_loader.var_get('instance_name'),
                              'user':ev_loader.var_get('user'),
                              'host':ev_loader.var_get('host'),
                              'service':ev_loader.var_get('service'),
                              'port':ev_loader.var_get('port'),
                              'password':ev_loader.var_get('password')}
        ConnectionPool.create_connection_pool(max_connections=1,
                                              connection_details=connection_details,
                                              logger=self.logger)
        #
        # self.db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
        #                                  user=ev_loader.var_get('user'),
        #                                  host=ev_loader.var_get('host'),
        #                                  service=ev_loader.var_get('service'),
        #                                  port=ev_loader.var_get('port'),
        #                                  password=ev_loader.var_get('password'),
        #                                  logger=self.logger)
        #
        self.spark = Spark(app_name=ev_loader.var_get('app_name'),
                           master=ev_loader.var_get('master'),
                           home_dir=ev_loader.var_get('home_dir'),
                           host_ip=ev_loader.var_get('host'),
                           spark_installation_path=ev_loader.var_get('spark_installation_path'),
                           spark_submit_deployMode=ev_loader.var_get('spark_submit_deployMode'),
                           spark_executor_instances=ev_loader.var_get('spark_executor_instances'),
                           spark_executor_memory=ev_loader.var_get('spark_executor_memory'),
                           spark_executor_cores=ev_loader.var_get('spark_executor_cores'),
                           spark_max_result_size=ev_loader.var_get('spark_max_result_size'),
                           spark_cores_max=ev_loader.var_get('spark_cores_max'),
                           spark_driver_memory=ev_loader.var_get('spark_driver_memory'),
                           spark_default_parallelism=ev_loader.var_get('spark_default_parallelism'),
                           spark_shuffle_partitions=ev_loader.var_get('spark_shuffle_partitions'),
                           spark_logConf=ev_loader.var_get('spark_logConf'),
                           spark_python_worker_reuse=ev_loader.var_get('spark_python_worker_reuse'),
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
    def initialize_spark(self):
        """
        Spark Context Initialization
        :return:
        """
        return self.spark
    #
    def get_global_config(self):
        return self.ev_loader