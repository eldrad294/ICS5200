#
# Module Imports
import time
#
class LoadTPCData:
    """
    This class contains all mapping functions which are utilized throughout this project.
    """
    #
    @staticmethod
    def send_partition(data_line, table_name, logger_details, instance_details):
        """
        Ships partition to slave executor, formats insert statements and executes them in parallel
        :param line: Current .DAT line
        :param table: Table data being loaded into
        :param instance_details: List containing instance details
        :return:
        """
        from src.framework.db_interface import DatabaseInterface
        from src.framework.logger import Logger
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
        #
        # Iterate over RDD partition
        row_count = 0
        for data in data_line:
            l_line = LoadTPCData.__parse_data_line(dataline=data)
            dml = "INSERT INTO " + table_name + " VALUES ("
            for i in range(len(l_line)):
                if i == 0:
                    dml += " :" + str(i+1) + " "
                else:
                    dml += ", :" + str(i+1) + " "
            dml += ")"
            di.execute_dml(dml, l_line)
            row_count += 1
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
                    value = int(value)
                except Exception:
                    try:
                        value = float(value)
                    except Exception:
                        pass
                #
                list_line.append(value)
                value = ""
        return tuple(list_line)
