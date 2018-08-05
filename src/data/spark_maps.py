class SparkMaps:
    """
    This class contains all mapping functions which are utilized throughout this project.
    """
    #
    @staticmethod
    def build_insert(dataline, table_name, database_context):
        """
        Formats insert statement
        :param line: Current .DAT line
        :param table: Table data being loaded into
        :param db_conn: Database connection context
        :param spark_context: Spark connection context
        :return:
        """
        print("Dataline")
        print(dataline)
        print("Tablename")
        print(table_name)
        print("Database Context")
        print(database_context)
        l_line = SparkMaps.__parse_data_line(dataline=dataline)
        dml = "INSERT INTO " + table_name + " VALUES ("
        for i in range(len(l_line)):
            if i == 0:
                dml += " :" + str(i+1) + " "
            else:
                dml += ", :" + str(i+1) + " "
        dml += ")"
        print(dml)
        database_context.execute_dml(dml, l_line).commit()
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
