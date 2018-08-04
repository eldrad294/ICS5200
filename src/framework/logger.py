#
# Module Imports
import datetime, time, os
#
class Logger:
    """
    This class acts under the the singleton design pattern, used for environment logging
    """
    #####################
    ## Private Members ##
    __instance = None
    __log_file_path = None
    __write_to_disk = None
    __write_to_screen = None
    #####################
    #
    #
    def __init__(self):
        if Logger.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self
    #
    @staticmethod
    def getInstance(log_file_path, write_to_disk, write_to_screen):
        """
        Invokes singleton instance
        :return: Singleton
                """
        if Logger.__instance is None:
            Logger()
            Logger.__log_file_path = log_file_path
            Logger.__write_to_disk = write_to_disk.title()
            Logger.__write_to_screen = write_to_screen.title()
        return Logger.__instance
    #
    @staticmethod
    def getTimeStamp():
        """
        Returns system timestamp
        :param self:
        :return:
        """
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    #
    @staticmethod
    def getDate():
        return datetime.datetime.today().strftime('%Y%m%d')
    #
    @staticmethod
    def log(msg):
        """
        Method invoked to either log to disk, or log to screen, or both
        :param msg: Message output/written to disk
        :return:
        """
        if Logger.__write_to_disk == 'True':
            try:
                if not os.path.exists(os.path.dirname(Logger.__log_file_path)):
                    os.makedirs(os.path.dirname(Logger.__log_file_path))
                #
                with open(Logger.__log_file_path,"a+") as myfile:
                    myfile.write(str(Logger.getTimeStamp()) + ": " + str(msg) + "\n")
            except OSError as ioe:
                raise OSError("An exception was raised during handling of log file [" + str(ioe) + "]")
        #
        if Logger.__write_to_screen == 'True':
            print(str(Logger.getTimeStamp()) + ": " + str(msg))
