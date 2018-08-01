#
# Module Imports
from src.framework.logger import logger
from src.framework.config_parser import g_config
from src.framework.env_var_loader import ev_loader
from os.path import expanduser
import os
home = expanduser("~")
#
class TPC_Wrapper:
    #
    __data_generated_directory = str(g_config.get_value('DataGeneration','data_generated_directory')) # Determines in which directory datafiles are generated
    __sql_generated_directory = str(g_config.get_value('DataGeneration','sql_generated_directory')) # Determines in which directory TPC SQL are generated
    __parallel_degree = int(g_config.get_value('DataGeneration', 'parallel_degree')) # Determines number of cores invoked to generate data with
    __data_size = int(g_config.get_value('DataGeneration', 'data_size')) # Determines size (in Gigabytes) of generated data
    __supported_tpc_types = ('TPC-DS', 'TPC-E')
    #
    @staticmethod
    def generate_data(tpc_type=None):
        """
        Method used to invoke respective TPC (TPC-DS/TPC-E) data generation tools
        :param tpc_type: Triggers either TPC-E or TPC-DS logic
        :return: None
        """
        #
        # Input validation
        TPC_Wrapper.__validate_input(tpc_type=tpc_type)
        #
        # Invoke respective TPC tool
        if tpc_type == TPC_Wrapper.__supported_tpc_types[0]:
            # TPC-DS
            dsdgen = ev_loader.var_get('project_dir') + "/data/TPC-DS/tools"
            #
            # Navigates to tool directory
            if not os.path.exists(TPC_Wrapper.__data_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0]):
                os.makedirs(TPC_Wrapper.__data_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0])
            os.chdir(dsdgen)
            #
            if TPC_Wrapper.__parallel_degree > 0:
                sys = "./dsdgen -scale " + str(TPC_Wrapper.__data_size) + " -dir " + TPC_Wrapper.__data_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0] + " -FORCE"
            elif TPC_Wrapper.__parallel_degree > 1:
                sys = "./dsdgen -f -scale " + str(TPC_Wrapper.__data_size) + " -dir " + TPC_Wrapper.__data_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0] + " -parallel " + str(TPC_Wrapper.__parallel_degree) + " -FORCE"
            else:
                raise Exception("Parallel degree not supported!")
            output = os.system(sys)
            if output != 0:
                raise Exception("Terminating process!")
            #
            logger.log(TPC_Wrapper.__supported_tpc_types[0] + " data generated for [" + str(TPC_Wrapper.__data_size) + "] Gigabytes using parallel degree [" + str(TPC_Wrapper.__parallel_degree) + "]")
        elif tpc_type == TPC_Wrapper.__supported_tpc_types[1]:
            raise NotImplementedError("TPC-E not supported yet!")
    #
    @staticmethod
    def generate_sql(tpc_type=None):
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
        TPC_Wrapper.__validate_input(tpc_type=tpc_type)
        #
        if tpc_type == TPC_Wrapper.__supported_tpc_types[0]:
            # TPC-DS - Query Generation
            dsqgen = ev_loader.var_get('project_dir')+"/data/TPC-DS/tools"
            #
            # Navigates to tool directory so as to invoke DSQGEN
            if not os.path.exists(TPC_Wrapper.__sql_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0] + "/Query"):
                os.makedirs(TPC_Wrapper.__sql_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0] + "/Query")
            os.chdir(dsqgen)
            #
            sys = "./dsqgen -DIRECTORY " + ev_loader.var_get('project_dir') + "/data/TPC-DS/query_templates -INPUT " + \
                  ev_loader.var_get('project_dir') + "/data/TPC-DS/query_templates/templates.lst -VERBOSE Y -QUALIFY Y " \
                  "-SCALE " + str(TPC_Wrapper.__parallel_degree) + " -DIALECT oracle -OUTPUT " + \
                  ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/Query"
            output = os.system(sys)
            if output != 0:
                raise Exception("An exception arose during dsqgen invocation..terminating process!")
                #
            logger.log(TPC_Wrapper.__supported_tpc_types[0] + " SQLs generated for dataset of [" + str(
                    TPC_Wrapper.__data_size) + "] Gigabytes")
            #
            # TPC-DS - DML Generation
            dml_data = ev_loader.var_get('project_dir')+"/data/TPC-DS/tests"
            dml_src = ev_loader.var_get('src_dir')+"/sql/Runtime/" + TPC_Wrapper.__supported_tpc_types[0] + "/DML"
            #
            if not os.path.exists(TPC_Wrapper.__sql_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0] + "/Query"):
                os.makedirs(TPC_Wrapper.__sql_generated_directory + "/" + TPC_Wrapper.__supported_tpc_types[0] + "/Query")
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
                    logger.log("Successfully migrated " + str(script) + " under src/ tree..")
            #
            logger.log("Successfully finished migrating DML scripts!")
        elif tpc_type == TPC_Wrapper.__supported_tpc_types[1]:
            raise NotImplementedError("TPC-E not supported yet!")
    #
    @staticmethod
    def split_tpc_sql_file(tpc_type=None):
        """
        Parses TPC query_0.sql, composed of all TPC sql in a single sql file. Place each respective sql in a separate
        SQL file for better handling.
        :param tpc_type:
        :return:
        """
        #
        # Input validation
        TPC_Wrapper.__validate_input(tpc_type=tpc_type)
        #
        query0_path = ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/Query/query_0.sql"
        #
        print(query0_path)
        print(os.path.exists(query0_path))
        if os.path.exists(query0_path) is False:
            raise FileNotFoundError('Query_0.sql was not found! Ensure that schema type ['+tpc_type+'] SQL generation has occurred!')
        else:
            #
            logger.log("Starting " + str(tpc_type) + " sql splitting")
            #
            # Read file into memory
            with open(query0_path) as f:
                read_data = f.read()
            #
            read_data = read_data.replace("\n","")
            #
            sql_list = read_data.split(";")
            for i, sql in enumerate(sql_list):
                with open(ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/Query/query_"+str(i+1)+".sql", "w") as f:
                    f.write(sql+";")
                logger.log("Generated query_" + str(i+1) + ".sql")
    #
    @staticmethod
    def __validate_input(tpc_type=None):
        """
        Validates that all input is configured as expected
        :param tpc_type:
        :return:
        """
        if tpc_type is None:
            raise Exception("TPC type not declared!")
        elif tpc_type not in TPC_Wrapper.__supported_tpc_types:
            raise Exception("TPC type not supported!")
        if TPC_Wrapper.__data_generated_directory is None:
            raise Exception("No target data directory was declared!")
    #
    @staticmethod
    def get_data_file_list(tpc_type=None):
        """
        Returns list of all generated data files
        :param tpc_type: Triggers either TPC-E or TPC-DS logic
        :return:
        """
        #
        # Input validation
        TPC_Wrapper.__validate_input(tpc_type=tpc_type)
        #
        file_list = os.listdir(TPC_Wrapper.__data_generated_directory + "/" + tpc_type)
        if file_list is None or len(file_list) < 1:
            raise Exception("No data files where found!")
        #
        return file_list
    #
    @staticmethod
    def get_file_extension_list(tpc_type=None):
        """
        Iterates over all available files, and retrieves each respective file name + extension
        :param tpc_type:
        :return:
        """
        file_names = []
        file_extensions = []
        for data_file_name in TPC_Wrapper.get_data_file_list(tpc_type=tpc_type):
            datum = data_file_name.split(".")
            if datum[0] is None or datum[1] is None or datum[0] == "" or datum[1] == "":
                raise ValueError("An attempt was made to parse non eligible files!")
            file_names.append(datum[0])
            file_extensions.append(datum[1])
        if len(file_names) != len(file_extensions):
            raise ValueError("File name list does not match list of file extensions!")
        return file_names, file_extensions
#
"""
Follow below example:
---------------------
TPC_Wrapper.generate_data(tpc_type='TPC-DS',
                          data_generated_directory=data_generated_directory,
                          data_size=1,
                          parallel_degree=2)
#
TPC_Wrapper.generate_sql(tpc_type='TPC-DS')
"""
