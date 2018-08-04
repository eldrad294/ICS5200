class FileLoaderUtils:
    """
    Contains static methods for calling of FileLoader context
    """
    __delimeter = '|'
    #
    @staticmethod
    def build_insert(line, table_name):
        """
        Formats insert statement
        :param line:
        :param table:
        :return:
        """
        l_line = FileLoaderUtils.__parse_data_line(line)
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
            if i != FileLoaderUtils.__delimeter:
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