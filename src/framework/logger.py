#
# Module Imports
import datetime, time, os
#
class Logger:
    """
    This class serves to log script messages.

    The core method is the 'log' method, allowing two modes of functionality:
    1) Log to disk (log files)
    2) Log to screen (log to screen)
    """
    #
    def __init__(self, log_file_path, log_script_name, log_user, write_to_disk, write_to_screen):
        self.__log_file_path = str(log_file_path) + log_script_name + "_" + log_user + "_" + self.__getDate()
        self.__write_to_disk = str(write_to_disk).title()
        self.__write_to_screen = str(write_to_screen).title()
        self.__del_logs() # Clean prior log generations
    #
    def __getTimeStamp(self):
        """
        :return: Returns system timestamp
        """
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    #
    def __getDate(self):
        """
        :return: Returns system date
        """
        return datetime.datetime.today().strftime('%Y%m%d')
    #
    def __del_logs(self):
        """
        Deletes all active logs
        :return:
        """
        del_cmd = 'rm -f ' + self.__log_file_path + '/msg_log* nohup_output*'
        output = os.system(del_cmd)
        if output != 0:
            raise Exception("Terminating process!")
    #
    def log(self, msg):
        """
        Method invoked to either log to disk, or log to screen, or both
        :param msg: Message output/written to disk
        :return:
        """
        if self.__write_to_disk == 'True':
            try:
                if not os.path.exists(os.path.dirname(self.__log_file_path)):
                    os.makedirs(os.path.dirname(self.__log_file_path))
                #
                with open(self.__log_file_path,"a+") as myfile:
                    myfile.write(str(self.__getTimeStamp()) + ": " + str(msg) + "\n")
            except OSError as ioe:
                raise OSError("An exception was raised during handling of log file [" + str(ioe) + "]")
        #
        if self.__write_to_screen == 'True':
            print(str(self.__getTimeStamp()) + ": " + str(msg))
