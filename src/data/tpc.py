#
# Module Imports
from src.framework.logger import logger
from src.framework.config_parser import g_config
from os.path import expanduser
import os
home = expanduser("~")
#
class TPC_Wrapper:
    #
    __data_generated_directory = str(g_config.get_value('DataGeneration','data_generated_directory')) # Determines in which directory datafiles are generated
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
            dsdgen = home+"/ICS5200/data/TPC-DS/tools"
            #
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
"""
