#
# Module Imports
from src.utils.config_parser import g_config
from src.utils.logger import logger
import cx_Oracle
import getpass
#
class DatabaseInterface:
    #
    def __init__(self, instance_name=None, user=None, host=None, service=None, port=None):
        self.__instance_name = instance_name
        self.__user = str(user)
        self.__host = str(host)
        self.__service = str(service)
        self.__port = str(port)
        self.__password = str(getpass.getpass("Enter database password:"))
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
    #
    def connect(self):
        """
        Establishes instance connection to Oracle database
        :return:
        """
        conn_str = self.__user + "/" + self.__password + "@" + self.__host + ":" + self.__port + "/" + self.__service
        try:
            self.conn = cx_Oracle.connect(conn_str)
            logger.log("Connected to database [" + self.__instance_name + "] with user [" + self.__user + "]")
        except Exception as e:
            logger.log("Exception caught whilst establishing connection to database! [" + str(e) + "]")
    #
    def query(self, query, params=None):
        """
        Statement wrapper method, invoked to pass query statements to the connected database instance
        :param query:
        :param params:
        :return:
        """
        cursor = self.conn.cursor()
        result = cursor.execute(query, params).fetchall()
        cursor.close()
        return result
    #
    def close(self):
        """
        Closes instance connection to Oracle database
        :return:
        """
        self.conn.close()
        logger.log("Connection closed to database [" + self.__instance_name + "] with user [" + self.__user + "]")
#
# Retrieves config data
instance_name = g_config.get_value('DatabaseConnectionString','instance_name')
user = g_config.get_value('DatabaseConnectionString','user')
host = g_config.get_value('DatabaseConnectionString','host')
service = g_config.get_value('DatabaseConnectionString','service')
port = g_config.get_value('DatabaseConnectionString','port')
#
db_conn = DatabaseInterface(instance_name=instance_name,
                            user=user,
                            host=host,
                            service=service,
                            port=port)

