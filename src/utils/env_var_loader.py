class EnvVarLoader():
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
ev_loader = EnvVarLoader.getInstance()


