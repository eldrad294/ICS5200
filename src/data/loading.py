#
# Module Imports
from src.data.spark_maps import LoadTPCData
#
class FileLoader:
    """
    This class is reserved for the un/loading of data into the database instance. The class is particularly customized
    for TPC generated data, allowing .dat files to be parsed, and loaded into database tables. It utilizes the Spark
    toolset to manipulate file un/loading in an efficient manner.
    """
    def __init__(self, ev_loader, logger, spark_context):
        self.__ev_loader = ev_loader
        self.__logger = logger
        self.__spark_context = spark_context
    #
    def load_data(self, path, table_name):
        """
        Loads data into memory using Spark RDDs, and inserts into Oracle DB
        :param path: data file path
        :param table_name: table/file name to be inserted inti / loaded from
        :return:
        """
        #
        # Materializes an RDD, but does not compute due to lazy evaluation
        rdd_file = self.__spark_context.textFile(path, self.__ev_loader.var_get('spark_cores_max') * 4)
        #
        # Pass database context details, to allow Spark executors to create their own connections
        instance_details = [self.__ev_loader.var_get('instance_name'),
                            self.__ev_loader.var_get('user'),
                            self.__ev_loader.var_get('host'),
                            self.__ev_loader.var_get('service'),
                            self.__ev_loader.var_get('port'),
                            self.__ev_loader.var_get('password')]
        #
        # Pass logger context details, to allow Spark executors to create their own logger context
        logger_details = [self.__ev_loader.var_get('log_file_path'),
                          self.__ev_loader.var_get('write_to_disk'),
                          self.__ev_loader.var_get('write_to_screen')]
        #
        # Pass Oracle path config, to allow Spark executors to make reference to database instance
        oracle_path_details = [self.__ev_loader.var_get('oracle_home'),
                               self.__ev_loader.var_get('ld_library_path')]
        #
        # Carry out Spark action on established RDDs
        rdd_file.foreachPartition(lambda iter, : LoadTPCData.send_partition(data=iter,
                                                                            table_name=table_name,
                                                                            logger_details=logger_details,
                                                                            instance_details=instance_details,
                                                                            oracle_path_details=oracle_path_details))
