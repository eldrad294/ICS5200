#
# Module Imports
import cx_Oracle
#
class DatabaseInterface:
    #
    def __init__(self, instance_name="gabsam", user="tpcds", host='192.168.202.222', service=None, port='1521'):
        self.__instance_name = instance_name
        self.__user = user
        self.__host = host
        self.__service = service
        self.__port = port
        self.__password = input("Enter database password:")
    #
    def connect(self):
        conn_str = self.__user + "/" + self.__password + "@" + self.__host + ":" + self.__port + "/" + self.__service
        self.conn = cx_Oracle.connect(conn_str)
        print("Connected to database [" + self.__instance_name + "] with user [" + self.__user + "]")
    #
    def query(self, query, params=None):
        cursor = self.conn.cursor()
        result = cursor.execute(query, params).fetchall()
        cursor.close()
        return result
    #
    def close(self):
        self.conn.close()
#
di = DatabaseInterface(instance_name="gabsam",
                       user="tpcds",
                       host="192.168.202.222",
                       service=None,
                       port="1521")
di.connect()

