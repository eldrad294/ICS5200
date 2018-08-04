#
# Module Imports
import os
import configparser
#
class ConfigParser:
    """
    Class responsible for parsing the main configuration file.
    """
    #################
    # Private members
    __config = None
    #################

    def __init__(self, config_file_path):
        """
        Constructor.
        :param config_file_path: Absolute path to the configuration file. Config file is expected to follow the
                                 standard INI file format.
        """
        self.reload_config_file(config_file_path)

    def reload_config_file(self, config_file_path):
        """
        Reload configuration file.
        :param config_file_path: Absolute path to the configuration file. Config file is expected to follow the
                                 standard INI file format.
        :return:                 None.
        """
        self.__validation(config_file_path)
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file_path)

    def get_value(self, section_name, key_name):
        """
        Retrieve configuration value.
        :param section_name: Config section name.
        :param key_name:     Config key name.
        :return:             Config value.
        """
        result = None
        if section_name in self.__config and key_name in self.__config[section_name]:
            result = self.__config[section_name][key_name]
        return result

    @staticmethod
    def __validation(config_file_path):
        """
        Validation.
        :param config_file_path: Absolute path to the configuration file.
        :return:                 None.
        :raises                  ValueError in case config file does not exist.
        """
        if not os.path.isfile(config_file_path):
            raise ValueError('File not found.')
#
class EnvVarLoader:
    """
    This class acts under the the singleton design pattern, used for environment variable loading + referencing
    """
    #####################
    ## Private Members ##
    __instance = None
    __env_vars = {}
    #####################
    #
    def __init__(self):
        """
        Virtual constructor, ensuring that the class behaves under the Singleton design pattern
        """
        if EnvVarLoader.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            EnvVarLoader.__instance = self
    #
    @staticmethod
    def getInstance():
        """
        Invokes singleton instance
        :return: Singleton
        """
        if EnvVarLoader.__instance is None:
            EnvVarLoader()
        return EnvVarLoader.__instance
    #
    @staticmethod
    def var_load(var_dict):
        """
        Loads variables inside singleton through a dictionary
        :param var_dict: key, value pairs pertaining to variable name + value
        :return:
        """
        for key, val in var_dict.items():
            if key not in EnvVarLoader.__env_vars:
                EnvVarLoader.__env_vars[key] = val
            else:
                raise Exception("Variable [" + str(key) + "] already loaded!")
    #
    @staticmethod
    def var_get(var_name=None):
        """
        Invokes Singleton dictionary and returns variable by input key.
        If key is not passed as param, return all dictionary.
        :param var_name:
        :return:
        """
        if var_name is None:
            return EnvVarLoader.__env_vars # Returns entire dictionary
        elif var_name not in EnvVarLoader.__env_vars:
            raise LookupError("Environment variable name not found!")
        return EnvVarLoader.__env_vars[var_name] # Returns variable value
#
# Defines env var object
ev_loader = EnvVarLoader.getInstance()


