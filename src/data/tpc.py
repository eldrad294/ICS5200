#
# Module Imports
from os.path import expanduser
from src.data.spark_maps import SparkMaps
import os
home = expanduser("~")
#
class TPC_Wrapper:
    #
    def __init__(self,
                 ev_loader,
                 logger,
                 database_context):
        self.__ev_loader = ev_loader
        self.__logger = logger
        self.__database_context = database_context
        self.__supported_tpc_types = ('TPC-DS', 'TPC-E')
    #
    def generate_data(self, tpc_type=None):
        """
        Method used to invoke respective TPC (TPC-DS/TPC-E) data generation tools
        :param tpc_type: Triggers either TPC-E or TPC-DS logic
        :return: None
        """
        #
        # Input validation
        self.__validate_input(tpc_type=tpc_type)
        #
        # Invoke respective TPC tool
        if tpc_type == self.__supported_tpc_types[0]:
            # TPC-DS
            dsdgen = self.__ev_loader.var_get('project_dir') + "/data/TPC-DS/tools"
            #
            # Navigates to tool directory
            data_generated_path = self.__ev_loader.var_get('data_generated_directory') + "/" + \
                                  self.__supported_tpc_types[0] + "/" + self.__ev_loader.var_get("user")
            if not os.path.exists(data_generated_path):
                os.makedirs(data_generated_path)
            os.chdir(dsdgen)
            #
            if self.__ev_loader.var_get('parallel_degree') > 0:
                sys = "./dsdgen -scale " + str(self.__ev_loader.var_get('data_size')) + " -dir " + data_generated_path + " -FORCE"
            elif self.__ev_loader.var_get('parallel_degree') > 1:
                sys = "./dsdgen -f -scale " + str(self.__ev_loader.var_get('data_size')) + " -dir " + data_generated_path \
                      + " -parallel " + str(self.__ev_loader.var_get('parallel_degree')) + " -FORCE"
            else:
                raise Exception("Parallel degree not supported!")
            self.__logger.log("Generating " + self.__supported_tpc_types[0] + " data for volumes of [" +
                       str(self.__ev_loader.var_get('data_size')) + "]..")
            output = os.system(sys)
            if output != 0:
                raise Exception("Terminating process!")
            #
            self.__logger.log(self.__supported_tpc_types[0] + " data generated for [" + str(self.__ev_loader.var_get('data_size'))
                       + "] Gigabytes using parallel degree [" + str(self.__ev_loader.var_get('parallel_degree')) + "]")
        elif tpc_type == self.__supported_tpc_types[1]:
            raise NotImplementedError("TPC-E not supported yet!")
    #
    def generate_sql(self, tpc_type=None):
        """
        Method used to invoke respective TPC (TPC-DS/TPC-E) SQL generation tools.
        The method is composed of dual functionality as follows:
        1) Invokes the DSQGEN tool to create 'query_0.sql' (which will later be segmented into a number of seperate SQL
           files) under the src/ directory
        2) Moves Data Maintenance tasks (DML Logic) and moves it under the src directory

        :param tpc_type: Triggers either TPC-E or TPC-DS logic
        :return: None
        """
        #
        # Input validation
        self.__validate_input(tpc_type=tpc_type)
        #
        if tpc_type == self.__supported_tpc_types[0]:
            # TPC-DS - Query Generation
            dsqgen = self.__ev_loader.var_get('project_dir')+"/data/TPC-DS/tools"
            #
            # Navigates to tool directory so as to invoke DSQGEN
            sql_generated_path = self.__ev_loader.var_get('sql_generated_directory') + "/" + self.__supported_tpc_types[0] + "/" + self.__ev_loader.var_get("user") + "/Query"
            if not os.path.exists(sql_generated_path):
                os.makedirs(sql_generated_path)
            os.chdir(dsqgen)
            #
            sys = "./dsqgen -DIRECTORY " + self.__ev_loader.var_get('project_dir') + "/data/TPC-DS/query_templates -INPUT " + \
                  self.__ev_loader.var_get('project_dir') + "/data/TPC-DS/query_templates/templates.lst -VERBOSE Y -QUALIFY Y " \
                  "-SCALE " + str(self.__ev_loader.var_get('parallel_degree')) + " -DIALECT oracle -OUTPUT " + sql_generated_path
            output = os.system(sys)
            if output != 0:
                raise Exception("An exception arose during dsqgen invocation..terminating process!")
                #
            self.__logger.log(self.__supported_tpc_types[0] + " SQLs generated for dataset of [" + str(
                self.__ev_loader.var_get('data_size')) + "] Gigabytes")
            #
            # TPC-DS - DML Generation
            dml_data = self.__ev_loader.var_get('project_dir')+"/data/TPC-DS/tests"
            dml_src = self.__ev_loader.var_get('src_dir')+"/sql/Runtime/" + self.__supported_tpc_types[0] + "/" + self.__ev_loader.var_get("user") + "/DML/"
            #
            if not os.path.exists(dml_src):
                os.makedirs(dml_src)
            os.chdir(dml_data)
            #
            target_scripts = [] # Keeps reference of which DML scripts to move under src/
            for filename in os.listdir(dml_data):
                if filename.endswith(".sql"):
                    target_scripts.append(filename)
            #
            for script in target_scripts:
                cmd = "cp " + dml_data + "/" + script + " " + dml_src
                output = os.system(cmd)
                if output != 0:
                    raise Exception("An exception arose during DML script migrations..terminating process!")
                else:
                    self.__logger.log("Successfully migrated " + str(script) + " under src/ tree..")
            #
            self.__logger.log("Successfully finished migrating DML scripts!")
        elif tpc_type == self.__supported_tpc_types[1]:
            raise NotImplementedError("TPC-E not supported yet!")
    #
    def split_tpc_sql_file(self, tpc_type=None):
        """
        Parses TPC query_0.sql, composed of all TPC sql in a single sql file. Place each respective sql in a separate
        SQL file for better handling.
        :param tpc_type:
        :return:
        """
        #
        # Input validation
        self.__validate_input(tpc_type=tpc_type)
        #
        query0_path = self.__ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/" + self.__ev_loader.var_get("user") + "/Query/query_0.sql"
        #
        if os.path.exists(query0_path) is False:
            raise FileNotFoundError('Query_0.sql was not found! Ensure that schema type ['+tpc_type+'] SQL generation has occurred!')
        else:
            #
            self.__logger.log("Starting " + str(tpc_type) + " sql splitting")
            #
            # Read file into memory
            with open(query0_path) as f:
                read_data = f.read()
            #
            read_data = read_data.replace("\n","")
            #
            sql_list = read_data.split(";")
            for i, sql in enumerate(sql_list):
                with open(self.__ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/" + self.__ev_loader.var_get("user") + "/Query/query_"+str(i+1)+".sql", "w") as f:
                    f.write(sql+";")
                self.__logger.log("Generated query_" + str(i+1) + ".sql")
    #
    def __validate_input(self, tpc_type=None):
        """
        Validates that all input is configured as expected
        :param tpc_type:
        :return:
        """
        if tpc_type is None:
            raise Exception("TPC type not declared!")
        elif tpc_type not in self.__supported_tpc_types:
            raise Exception("TPC type not supported!")
        if self.__ev_loader.var_get('data_generated_directory') is None:
            raise Exception("No target data directory was declared!")
    #
    def get_data_file_list(self, tpc_type=None):
        """
        Returns list of all generated data files
        :param tpc_type: Triggers either TPC-E or TPC-DS logic
        :return:
        """
        #
        # Input validation
        self.__validate_input(tpc_type=tpc_type)
        #
        file_list = os.listdir(self.__ev_loader.var_get('data_generated_directory') + "/" + tpc_type + "/" + self.__ev_loader.var_get('user'))
        if file_list is None or len(file_list) < 1:
            raise Exception("No data files where found!")
        #
        return file_list
    #
    def get_file_extension_list(self, tpc_type=None):
        """
        Iterates over all available files, and retrieves each respective file name + extension
        :param tpc_type:
        :return:
        """
        file_names = []
        file_extensions = []
        for data_file_name in self.get_data_file_list(tpc_type=tpc_type):
            datum = data_file_name.split(".")
            if datum[0] is None or datum[1] is None or datum[0] == "" or datum[1] == "":
                raise ValueError("An attempt was made to parse non eligible files!")
            file_names.append(datum[0])
            file_extensions.append(datum[1])
        if len(file_names) != len(file_extensions):
            raise ValueError("File name list does not match list of file extensions!")
        return file_names, file_extensions
#
class FileLoader:
    """
    This class is reserved for the un/loading of data into the database instance. The class is particularly customized
    for TPC generated data, allowing .dat files to be parsed, and loaded into database tables. It utilizes the Spark
    toolset to manipulate file un/loading in an efficient manner.
    """
    def __init__(self, ev_loader, logger, database_context, spark_context):
        self.__ev_loader = ev_loader
        self.__logger = logger
        self.__database_context = database_context
        self.__spark_context = spark_context

    def load_data(self, path, table_name):
        """
        Loads data into memory using Spark RDDs, and inserts into Oracle DB
        :param path: data file path
        :param table_name: table/file name to be inserted inti / loaded from
        :param database_context: Oracle connection context
        :param spark_context: Spark connection context
        :return:
        """
        rdd_file = self.__spark_context.textFile(path, self.__ev_loader.var_get('spark_rdd_partitions')) # Materializes an RDD, but does not compute due to lazy evaluation
        mapped_rdd_file = rdd_file.map(lambda x: x.split('\n')) # Split line by line - does not compute immediately due to lazy evaluation
        db_conn = self.__database_context
        mapped_rdd_file.foreachPartition(lambda line : SparkMaps.build_insert(dataline=line,
                                                                       table_name=table_name,
                                                                       database_context=db_conn))
        #
        self.__logger.log("Loaded table [" + table_name + "] into database..")
