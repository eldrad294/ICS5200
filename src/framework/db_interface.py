#
# Module Imports
import cx_Oracle, time, os
#
class DatabaseInterface:
    #
    def __init__(self, instance_name=None, user=None, host=None, service=None, port=None, password=None, logger=None):
        self.__instance_name = str(instance_name)
        self.__user = str(user)
        self.__host = str(host)
        self.__service = str(service)
        self.__port = str(port)
        self.__password = str(password) # Required to execute under nohup instead of manual user input,
        self.__logger = logger
        self.__conn = None
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
        if self.__logger is None:
            raise ValueError("Database logger was not declared!")
        if self.__password is None:
            raise ValueError("Password was not declared!")
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
            self.__conn = cx_Oracle.connect(conn_str, encoding = "UTF-8", nencoding = "UTF-8")
        except Exception as e:
            self.__logger.log("Exception caught whilst establishing connection to database! [" + str(e) + "]")
            #raise Exception("Couldn't connect to database: [" + str(e) + "]")
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
        cursor, result, description = None, None, None
        try:
            cursor = self.__conn.cursor()
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
        Statement wrapper method invoked to pass dml statements to the connected database instance.
        Expected to return no results from query execution
        :param dml: (insert, update, delete, merge, explain plan for, etc...)
        :param params: dictionary of bind variables
        :return:
        """
        cursor = None
        try:
            cursor = self.__conn.cursor()
            if params is None:
                cursor.execute(dml)
            else:
                cursor.execute(dml, params)
        except Exception as e:
            if self.__logger is not None:
                self.__logger.log('Skipped DML instruction due to following exception: [' + str(e) + '] - Instruction: [' +
                        str(dml) + ' ]')
        finally:
            if cursor is not None:
                cursor.close()
    #
    def execute_many_dml(self, dml, data):
        cursor = None
        try:
            cursor = self.__conn.cursor()
            cursor.executemany(dml, data, batcherrors=True)

            #display the errors that have taken place
            errors = cursor.getbatcherrors()
            self.__logger.log("number of errors which took place:" + str(len(errors)))
            for error in errors:
                self.__logger.log("Error " + str(error.message.rstrip()) + " at row offset " + str(error.offset))
        except Exception as e:
            if self.__logger is not None:
                self.__logger.log(
                    'Skipped DML instruction due to following exception: [' + str(e) + '] - Instruction: [' +
                    str(dml) + ' ]')
        finally:
            if cursor is not None:
                cursor.close()
    #
    def execute_proc(self, name, parameters):
        """
        Executes procedure
        :param name: Name of procedure
        :param parameters: Dictionary of parameters
        :return:
        """
        cursor = None
        try:
            cursor = self.__conn.cursor()
            cursor.callproc(name=name, keywordParameters=parameters)
        except Exception as e:
            if self.__logger is not None:
                self.__logger.log(
                    'Skipped DML instruction due to following exception: [' + str(e) + '] - Instruction: [' +
                    str(name) + ' ] with params ' + str(parameters))
        finally:
            if cursor is not None:
                cursor.close()
    #
    def commit(self):
        """
        Commits transaction/s
        :return:
        """
        try:
            self.__conn.commit()
        except Exception as e:
            self.__logger.log("Couldn't commit transaction to database: [" + str(e) + "]")
            #raise Exception("Couldn't commit transaction to database: [" + str(e) + "]")
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
    def execute_script(self, user, password, instance_name, filename, params):
        sys = "exit | sqlplus " + user + "/" + password + "@" + instance_name + " @" + filename
        if params is not None and len(params) > 0:
            for param in params:
                sys += " " + str(param)
        self.__logger.log(sys)
        output = os.system(sys)
        if output != 0:
            raise Exception("Exception raised during generation of TPC files..Terminating process!")
        # self.__logger.log(sys)
        self.__logger.log(filename + " executed!")
    #
    def close(self):
        """
        Closes instance connection to Oracle database
        :return:
        """
        try:
            self.__conn.close()
            time.sleep(1)
        except Exception as e:
            self.__logger.log("Couldn't close connection: [" + str(e) + "]")
    #
    def get_connection_details(self):
        return {'instance_name':self.__instance_name,
                'user':self.__user,
                'host':self.__host,
                'service':self.__service,
                'port':self.__port}
#
class ConnectionPool:
    """
    Connection pool class
    """
    #
    __pool = []
    #
    def __init__(self):
        raise Exception('This class cannot be instantiated!')
    #
    @staticmethod
    def create_connection_pool(max_connections, connection_details, logger):
        max_connections = int(max_connections)
        if max_connections is None:
            raise ValueError('Maximum connection pool size must be declared!')
        if max_connections > 210 or max_connections < 1:
            raise ValueError('Connection pool size must be between 1 and 210!')
        if len(connection_details) == 0 or connection_details is None:
            raise ValueError('No connection details were specified!')
        #
        for i in range(max_connections):
            conn = DatabaseInterface(instance_name=connection_details['instance_name'],
                                     user=connection_details['user'],
                                     host=connection_details['host'],
                                     service=connection_details['service'],
                                     port=connection_details['port'],
                                     password=connection_details['password'],
                                     logger=logger)
            conn.connect()
            conn_list = [i,0,conn] # id, status {0,1}, connection
            ConnectionPool.__pool.append(conn_list)
        logger.log('Connection pool instantiated with [' + str(max_connections) + '] connections')
    #
    @staticmethod
    def close_connection_pool():
        for conn_list in ConnectionPool.__pool:
            if conn_list[1] == 1:
                conn_list[2].close()
    #
    @staticmethod
    def claim_from_pool():
        if len(ConnectionPool.__pool) == 0:
            raise Exception('Connection pool is empty!')
        #
        for i, conn_list in enumerate(ConnectionPool.__pool):
            status = conn_list[1]
            if status == 0:
                ConnectionPool.__pool[i][1] = 1
                return ConnectionPool.__pool[i] # Returns Connection List
        else:
            raise Exception('Connection pool busy..all [' + str(len(ConnectionPool.__pool)) + '] connections are currently active!')
    #
    @staticmethod
    def return_to_pool(conn):
        conn_list = ConnectionPool.__pool[id]
        if conn_list[conn[0]][1] == 1:
            conn.close()
            ConnectionPool.__pool[conn[0]][1] = 0
"""
Follow below example:
---------------------
db_conn.connect()
rec_cur = db_conn.execute_query('select 1 from dual')
print(rec_cur)
"""
