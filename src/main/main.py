#
# Module Imports
import sys
from os.path import dirname, abspath
#
# Retrieving relative paths for project directory
project_dir = dirname(dirname(dirname(abspath(__file__))))
src_dir = dirname(dirname(abspath(__file__)))
#
# Appending to python path
sys.path.append(project_dir)
sys.path.append(src_dir)
#
from src.utils.env_var_loader import ev_loader
#
# Loading of program variables
ev_loader.var_load({"project_dir":project_dir,"src_dir":src_dir})
"""
--------------IMPORTANT!!--------------
-------PLACE NEW MODULES BELOW!!-------
---------------------------------------
"""
from src.utils.logger import logger
from src.data.tpc import TPC_Wrapper
from src.utils.db_interface import db_conn
from src.utils.config_parser import g_config
from src.data.loading import FileLoader
#
# Establishes database connection
db_conn.connect()
#
tpcds_generation_bool, tpce_generation_bool = g_config.get_value('DataGeneration','tpcds_generation').title(), \
                                              g_config.get_value('DataGeneration','tpce_generation').title()
#
"""
Data Generation
"""
print(tpcds_generation_bool)
print(g_config.get_value('DataGeneration','tpcds_generation'))
if tpcds_generation_bool  == 'True':
    TPC_Wrapper.generate_data(tpc_type='TPC-DS')
if tpce_generation_bool == 'True':
    raise NotImplementedError("This logic is not yet implemented!")
#
"""
Data Loading
"""
tpcds_loading_bool, tpce_loading_bool = g_config.get_value('DataLoading','tpcds_loading').title(), \
                                        g_config.get_value('DataLoading','tpce_loading').title()
data_generated_dir = str(g_config.get_value('DataGeneration','data_generated_directory'))
fl = FileLoader(app_name="ICS5200", master="local")
if tpcds_loading_bool == 'True':
    #
    # Retrieve all eligible data files
    for data_file_name in TPC_Wrapper.get_data_file_list("TPC-DS"):
        fl.load_data(data_generated_dir + "/TPC-DS/" + data_file_name)
if tpce_loading_bool == 'True':
    raise NotImplementedError("This logic is not yet implemented!")
#
logger.log("Script Complete!\n-------------------------------------")
