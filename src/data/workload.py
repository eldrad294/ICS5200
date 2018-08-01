#
# Import Modules
from src.framework.logger import logger
from src.framework.config_parser import g_config
from src.utils.randomizer import RandomizerUtility
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
        Reads config from src/main/config.ini
        """
        self.workload_time_length = float(g_config.get_value('WorkloadGeneration','outer_workload_time_window'))
        self.num_sql = float(g_config.get_value('WorkloadGeneration','num_sql'))
        self.num_dml = float(g_config.get_value('WorkloadGeneration','num_dml'))
        self.min_interval, self.max_interval = self.__process_interval(g_config.get_value('WorkloadGeneration','interval'))
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
            min_interval = int(interval_list[0])
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