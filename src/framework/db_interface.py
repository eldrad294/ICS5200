#
# Module Imports
import cx_Oracle
import getpass
#
class DatabaseInterface:
    #
    def __init__(self, instance_name=None, user=None, host=None, service=None, port=None, password=None, logger=None):
        self.__instance_name = instance_name
        self.__user = str(user)
        self.__host = str(host)
        self.__service = str(service)
        self.__port = str(port)
        self.__password = str(password) # Required to execute under nohup instead of manual user input,
        self.__logger = logger
        #
        # Validates connection config
        self.__validate_db_config()
    #
    def __validate_db_config(self):
        """
        Validates connection string data, to validate that all required config was supplied
        :return:
        """
        if self.__instance_name is None:
            raise ValueError("Instance name was not declared!")
        if self.__user is None:
            raise ValueError("Database user was not declared!")
        if self.__host is None:
            raise ValueError("Database host was not declared!")
        if self.__service is None:
            raise ValueError("Database service was not declared!")
        if self.__port is None:
            raise ValueError("Database port was not declared!")
        # if self.__logger is None:
        #     raise ValueError("Logger context was not declared!")
    #
    def __clean_query(self, v_sql):
        return v_sql.replace("\n"," ")
    #
    def __schema_names(self, schema):
        """
        Returns CX_Oracle table description, and returns the table column names as a list
        :param schema:
        :return:
        """
        column_names = []
        #
        if schema is None:
            raise ValueError('Schema descriptor is empty!')
        #
        for element in schema:
            column_names.append(element[0])
        return column_names
    #
    def connect(self):
        """
        Establishes instance connection to Oracle database
        :return:
        """
        conn_str = self.__user + "/" + self.__password + "@" + self.__host + ":" + self.__port + "/" + self.__service
        try:
            self.conn = cx_Oracle.connect(conn_str, encoding = "UTF-8", nencoding = "UTF-8")
            if self.__logger is not None:
                self.__logger.log("Connected to database [" + self.__instance_name + "] with user [" + self.__user + "]")
            else:
                print("Connected to database [" + self.__instance_name + "] with user [" + self.__user + "]")
        except Exception as e:
            if self.__logger is not None:
                self.__logger.log("Exception caught whilst establishing connection to database! [" + str(e) + "]")
            else:
                print("Exception caught whilst establishing connection to database! [" + str(e) + "]")
    #
    def execute_query(self, query, params=None, fetch_single=False, describe=False):
        """
        Statement wrapper method, invoked to pass query statements to the connected database instance, and return
        cursor result set in the form of a tuple set.
        Expected to return results from query execution
        :param query: SQL statement (selects)
        :param params: dictionary of bind variables
        :param fetch_single: warns code logic that returned cursor will consist of a single result
        :param describe: Defines whether table description is also returned
        :return:
        """
        cursor = self.conn.cursor()
        result, description = None, None
        #query = self.__clean_query(query)
        try:
            if fetch_single is True:
                if params is None:
                    result = cursor.execute(query).fetchone()
                else:
                    result = cursor.execute(query, params).fetchone()
            else:
                if params is None:
                    result = cursor.execute(query).fetchall()
                else:
                    result = cursor.execute(query, params).fetchall()
            #
            if describe is True:
                description = cursor.description
        except Exception as e:
            if self.__logger is not None:
                self.__logger.log('Skipped record due to following exception: [' + str(e) + ']')
            else:
                print('Skipped record due to following exception: [' + str(e) + ']')
        finally:
            if cursor is not None:
                cursor.close()
        #
        if describe is True:
            return result, self.__schema_names(description)
        else:
            return result
    #
    def execute_dml(self, dml, params=None):
        """
        Statement wrapper methodm invokled to pass dml statements to the connected database instance.
        Expected to return no results from query execution
        :param dml: (insert, update, delete, merge, explain plan for, etc...)
        :param params: dictionary of bind variables
        :return:
        """
        cursor = self.conn.cursor()
        #dml = self.__clean_query(dml)
        try:
            if params is None:
                cursor.execute(dml)
            else:
                cursor.execute(dml, params)
        except Exception as e:
            if self.__logger is not None:
                self.__logger.log('Skipped DML instruction due to following exception: [' + str(e) + '] - Instruction: [' +
                        str(dml) + ' ]')
            else:
                print('Skipped DML instruction due to following exception: [' + str(e) + '] - Instruction: [' + str(dml) + ' ]')
        finally:
            if cursor is not None:
                cursor.close()
    #
    def commit(self):
        """
        Commits transaction/s
        :return:
        """
        self.conn.commit()
    #
    def executeScriptsFromFile(self, filename):
        """
        Opens SQL file, separates content by ';' delimiter, and executes each instruction.
        :return:
        """
        # Open and read the file as a single buffer
        fd = open(filename, 'r')
        sqlFile = fd.read()
        fd.close()
        #
        # all SQL commands (split on ';')
        sqlCommands = sqlFile.split(';')
        #
        # Execute every command from the input file
        for command in sqlCommands:
            # This will skip and report errors
            # For example, if the tables do not yet exist, this will skip over
            # the DROP TABLE commands
            if command is not None and command != "" and command != '\n\n':
                self.execute_dml(command)
    #
    def close(self):
        """
        Closes instance connection to Oracle database
        :return:
        """
        self.conn.close()
        if self.__logger is not None:
            self.__logger.log("Connection closed to database [" + self.__instance_name + "] with user [" + self.__user + "]")
        else:
            print("Connection closed to database [" + self.__instance_name + "] with user [" + self.__user + "]")
    #
    def get_connection_details(self):
        return {'instance_name':self.__instance_name,
                'user':self.__user,
                'host':self.__host,
                'service':self.__service,
                'port':self.__port}
"""
Follow below example:
---------------------
db_conn.connect()
rec_cur = db_conn.execute_query('select 1 from dual')
print(rec_cur)
"""
