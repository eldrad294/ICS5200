#
# Import Modules
from src.framework.logger import logger
from src.framework.config_parser import g_config
import random
#
class Workload:
    """
    This class is dedicated to generating an artificial workload over TPC-esq schemas, mainly:
    * TPC-DS
    * TPC-E
    """
    #
    def __init__(self):
        """
        Reads config from src/main/config.ini, and convert into appropriate data types
        """
        self.__outer_workload_time_window = g_config.get_value('WorkloadGeneration','outer_workload_time_window')
        self.__inner_workload_time_window = g_config.get_value('WorkloadGeneration','inner_workload_time_window')
        self.__min_sql_sample_range, self.__max_sql_sample_range = self.__process_interval(g_config.get_value('WorkloadGeneration','sql_sample_range'))
        self.__min_dml_sample_range, self.__max_dml_sample_range = self.__process_interval(g_config.get_value('WorkloadGeneration', 'dml_sample_range'))
        self.__min_sql_sample, self.__max_sql_sample = self.__process_interval(g_config.get_value('WorkloadGeneration', 'sql_sample'))
        self.__min_num_dml, self.__max_num_dml = self.__process_interval(g_config.get_value('WorkloadGeneration', 'num_dml'))
        self.__min_outer_interval, self.__max_outer_interval = self.__process_interval(g_config.get_value('WorkloadGeneration','outer_interval'))
        self.__min_inner_interval, self.__max_inner_interval = self.__process_interval(g_config.get_value('WorkloadGeneration', 'inner_interval'))
        self.__repeats = g_config.get_value('WorkloadGeneration', 'repeats')
        #
        # Type cast to appropriate data types
        self.__cast_vars()
        #
        # Validate config
        self.__validate_vars()
        #
        # Randomize inputs based on seeded input
        self.__sql_sample_range = random.randint(self.__min_sql_sample_range, self.__max_sql_sample_range)
        self.__dml_sample_range = random.randint(self.__min_dml_sample_range, self.__max_dml_sample_range)
        self.__sql_sample = random.randint(self.__min_sql_sample, self.__max_sql_sample)
        self.__num_dml = random.randint(self.__min_num_dml, self.__max_num_dml)
        self.__outer_interval = random.randint(self.__min_outer_interval, self.__max_outer_interval)
        self.__inner_interval = random.randint(self.__min_inner_interval, self.__max_inner_interval)
        logger.log('Workload parameters successfully randomized from seeded input..')
    #
    def __cast_vars(self):
        """
        Type cast to appropriate data types
        :return:
        """
        # outer_workload_time_window
        self.__outer_workload_time_window = float(self.__outer_workload_time_window)
        # inner_workload_time_window
        self.__inner_workload_time_window = float(self.__inner_workload_time_window)
        # sql_sample_range
        self.__min_sql_sample_range = int(self.__min_sql_sample_range)
        self.__max_sql_sample_range = int(self.__max_sql_sample_range)
        # dml_sample_range
        self.__min_dml_sample_range = int(self.__min_dml_sample_range)
        self.__max_dml_sample_range = int(self.__max_dml_sample_range)
        # sql_sample
        self.__min_sql_sample = int(self.__min_sql_sample)
        self.__max_sql_sample = int(self.__max_sql_sample)
        # num_dml
        self.__min_num_dml = int(self.__min_num_dml)
        self.__max_num_dml = int(self.__max_num_dml)
        # outer_interval
        self.__min_outer_interval = float(self.__min_outer_interval)
        self.__max_outer_interval = float(self.__max_outer_interval)
        # inner_interval
        self.__min_inner_interval = float(self.__min_inner_interval)
        self.__max_inner_interval = float(self.__max_inner_interval)
        # repeats
        self.__repeats = int(self.__repeats)
    #
    def __validate_vars(self):
        """
        Validates file config
        :return:
        """
        if self.__outer_workload_time_window <= 0:
            raise ValueError('Outer workload time must be greater than 0!')
        if self.__inner_workload_time_window <= 0:
            raise ValueError('Inner workload time must be greater then 0!')
        if self.__min_sql_sample_range <= 0 or self.__max_sql_sample_range <= 0:
            raise ValueError('sql_sample_range values must be greater than 0!')
        if self.__min_dml_sample_range <= 0 or self.__max_dml_sample_range <= 0:
            raise ValueError('dml_sample_range values must be greater than 0!')
        if self.__min_sql_sample <= 0 or self.__max_sql_sample <= 0:
            raise ValueError('sql_sample values must be greater than 0!')
        if self.__min_num_dml <= 0 or self.__max_num_dml <= 0:
            raise ValueError('num_dml values must be greater than 0!')
        if self.__min_outer_interval <= 0 or self.__max_outer_interval <= 0:
            raise ValueError('outer_interval values must be greater than 0!')
        if self.__min_inner_interval <= 0 or self.__max_inner_interval <= 0:
            raise ValueError('inner_interval values must be greater than 0!')
        if self.__repeats <= 0:
            raise ValueError('repeats values must be greater than 0!')
        #
        if self.__min_sql_sample_range > self.__max_sql_sample_range:
            raise ValueError('Incorrect sql_sample_range. Min value must be smaller than max!')
        if self.__min_dml_sample_range > self.__max_dml_sample_range:
            raise ValueError('Incorrect dml_sample_range. Min value must be smaller than max!')
        if self.__min_sql_sample > self.__max_sql_sample:
            raise ValueError('Incorrect sql_sample. Min value must be smaller than max!')
        if self.__min_num_dml > self.__max_num_dml:
            raise ValueError('Incorrect num_dml. Min value must be smaller than max!')
        if self.__min_outer_interval > self.__max_outer_interval:
            raise ValueError('Incorrect outer_interval. Min value must be smaller than max!')
        if self.__min_inner_interval > self.__max_inner_interval:
            raise ValueError('Incorrect inner_interval. Min value must be smaller than max!')
    #
    def __process_interval(self, interval):
        """
        Accepts string of following format '0.05-0.1', and returns as two separate float variables
        :param interval:
        :return: min_interval, max_interval
        """
        interval_list = interval.split("-")
        if len(interval_list) != 2:
            raise ValueError("Interval parameter is not properly defined!")
        #
        try:
            min_interval = float(interval_list[0])
        except Exception:
            raise ValueError("An exception was raised during type casting of min_interval!")
        #
        try:
            max_interval = int(interval_list[1])
        except Exception:
            raise ValueError("An exception was raised during type casting of max_interval!")
        #
        if min_interval > max_interval:
            raise ValueError("Min interval must be defined a smaller value than max interval!")
        #
        return min_interval, max_interval
    #
    def execute_workload(self):
        pass