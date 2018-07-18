#
# Module Imports
from src.utils.logger import logger
from src.utils.config_parser import g_config
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
"""
Follow below example:
---------------------
TPC_Wrapper.generate_data(tpc_type='TPC-DS',
                          data_generated_directory=data_generated_directory,
                          data_size=1,
                          parallel_degree=2)
"""
