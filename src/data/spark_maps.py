class SparkMaps:
    """
    This class contains all mapping functions which are utilized throughout this project.
    """
    #
    @staticmethod
    def build_insert(dataline, table_name, database_context, ):
        """
        Formats insert statement
        :param line: Current .DAT line
        :param table: Table data being loaded into
        :param db_conn: Database connection context
        :param spark_context: Spark connection context
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
        database_context.execute_dml(dml, l_line).commit()
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
        delimeter = '|'
        value = ""
        for i in line:
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
