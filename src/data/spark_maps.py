#
# Module Imports
from src.framework.db_interface import DatabaseInterface
#
class SparkMaps:
    """
    This class contains all mapping functions which are utilized throughout this project.
    """
    #
    @staticmethod
    def send_partition(data, table_name, instance_details):
        """
        Ships partition to slave executor, formats insert statements and executes them in parallel
        :param line: Current .DAT line
        :param table: Table data being loaded into
        :return:
        """
        di = DatabaseInterface(instance_name=instance_details[0],
                               user=instance_details[1],
                               host=instance_details[2],
                               service=instance_details[3],
                               port=instance_details[4],
                               password=instance_details[5])
        di.connect()
        print(data)
        l_line = SparkMaps.__parse_data_line(dataline=data)
        print(l_line)
        dml = "INSERT INTO " + table_name + " VALUES ("
        for i in range(len(l_line)):
            if i == 0:
                dml += " :" + str(i+1) + " "
            else:
                dml += ", :" + str(i+1) + " "
        dml += ")"
        print(dml)
        di.execute_dml(dml, l_line)
        di.commit()
        di.close()
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
