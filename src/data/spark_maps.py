class SparkMaps:
    """
    This class contains all mapping functions which are utilized throughout this project.

    It is IMPERATIVE that this class remains a standalone class, and that it should AVOID any global variable
    references, including module imports outside the scope of this class but within the same .py class. Note, that this
    class is reserved to mapping functions - functions which will be executed on parallel executors, and will therefore
    not have reference to variables usually accessed by Spark's Main driver.
    """
    __delimeter = '|'
    #
    @staticmethod
    def build_insert(dataline, table_name, db_conn):
        """
        Formats insert statement
        :param line:
        :param table:
        :return:
        """
        l_line = SparkMaps.__parse_data_line(dataline)
        dml = "INSERT INTO " + table_name + " VALUES ("
        for i in range(len(l_line)):
            if i == 0:
                dml += " :" + str(i+1) + " "
            else:
                dml += ", :" + str(i+1) + " "
        dml += ")"
        db_conn.execute_dml(dml, l_line)
    #
    @staticmethod
    def __parse_data_line(line):
        """
        Iterates over input data line, and parses value into a list. Values are delimeted according to config file,
        default to '|'
        :param line:
        :return:
        """
        list_line = []
        value = ""
        for i in line:
            if i != SparkMaps.__delimeter:
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