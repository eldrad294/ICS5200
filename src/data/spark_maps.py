#
# Module Imports
from src.framework.db_interface import DatabaseInterface
from src.framework.logger import Logger
from decimal import Decimal
import time, os
#
class LoadTPCData:
    """
    This class contains all mapping functions which are utilized throughout this project.
    """
    #
    @staticmethod
    def send_partition(data, table_name, logger_details, instance_details, oracle_path_details):
        """
        Ships partition to slave executor, formats insert statements and executes them in parallel
        :param line: Current .DAT line
        :param table: Table data being loaded into
        :param instance_details: List containing instance details
        :param oracle_path_details: List of Oracle instance path + libs
        :return:
        """
        os.environ['ORACLE_HOME'] = oracle_path_details[0]
        os.environ['LD_LIBRARY_PATH'] = oracle_path_details[1]
        #
        start_time = time.time()
        #
        # Establish slave logger
        logger = Logger(log_file_path=logger_details[0],
                        write_to_disk=logger_details[1],
                        write_to_screen=logger_details[2])
        logger.log('Starting data migration into table [' + table_name + ']')
        #
        # Establish slave database context
        di = DatabaseInterface(instance_name=instance_details[0],
                               user=instance_details[1],
                               host=instance_details[2],
                               service=instance_details[3],
                               port=instance_details[4],
                               password=instance_details[5])
        di.connect()
        time.sleep(1) # back of time to allow time for connection to establish in case of very small data sets
        #
        # Retrieve columns required for batch insert
        sql = "select column_name from user_tab_columns where table_name = '" + table_name.upper() + "'";
        res = di.execute_query(query=sql, describe=False)
        column_names = "("
        for i, item in enumerate(res):
            if i == 0:
                column_names += str(item[0])
            else:
                column_names += ',' + str(item[0])
        else:
            column_names += ')'
        #
        # Iterate over RDD partition
        row_count = 0
        values_bank = []
        dml = "INSERT INTO " + table_name + " " + column_names + " VALUES ("
        for count, data_line in enumerate(data):
            l_line = LoadTPCData.__parse_data_line(dataline=data_line)
            if table_name == 'TIME_DIM' or table_name == 'time_dim':
                logger.log(l_line)
            if count < 1:
                for i in range(len(l_line)):
                    if i == 0:
                        dml += " :" + str(i+1) + " "
                    else:
                        dml += ", :" + str(i+1) + " "
                dml += ")"
            values_bank.append(l_line)
            row_count += 1
        di.execute_many_dml(dml=dml, data=values_bank) # Bulk Insert
        di.commit() # Commit once after every RDD batch
        di.close()
        #
        end_time = time.time()
        logger.log('Committed ' + str(row_count) + ' rows for table ' + table_name + " | " + str(end_time-start_time) + " seconds")
    #
    @staticmethod
    def __parse_data_line(dataline):
        """
        Iterates over input data line, and parses value into a list. Values are delimeted according to config file,
        default to '|'
        :param line:
        :return:
        """
        list_line = []
        delimeter = '|'
        value = ""
        for i in dataline:
            if i != delimeter:
                value += i
            else:
                try:
                    if Decimal(value) % 1 == 0:
                        list_line.append(int(value))
                    else:
                        list_line.append(float(value))
                except Exception:
                    list_line.append(str(value))
                value = ""
        return tuple(list_line)
