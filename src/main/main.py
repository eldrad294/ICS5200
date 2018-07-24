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
from src.framework.env_var_loader import ev_loader
#
# Loading of program variables
ev_loader.var_load({"project_dir":project_dir,"src_dir":src_dir})
"""
--------------IMPORTANT!!--------------
-------PLACE NEW MODULES BELOW!!-------
---------------------------------------
"""
from src.framework.logger import logger
from src.data.tpc import TPC_Wrapper
from src.framework.db_interface import db_conn
from src.framework.config_parser import g_config
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
#
fl = FileLoader(app_name="ICS5200", master="local")
if tpcds_loading_bool == 'True':
    #
    # Check whether schema needs creating - executed only if relevant tables are not found
    sql_statement = "select count(*) from user_tables where table_name = 'STORE_SALES'"
    result = int(db_conn.execute_query(sql_statement))
    print(result)
    if result < 1:
        print(ev_loader.var_get("src_dir") + "/sql/Installation/tpcds_schema_tables.sql")
        db_conn.executeScriptsFromFile(ev_loader.var_get("src_dir") + "/sql/Installation/tpcds_schema_tables.sql")
        logger.log('TPC-DS schema generation successful!')
    else:
        logger.log('Skipping schema creation..TPC-DS tables already exist!')
    #
    # Retrieve eligible data file names
    table_names = TPC_Wrapper.get_file_extension_list(tpc_type="TPC-DS")[0]
    #
    # Retrieve all eligible data files
    file_names = TPC_Wrapper.get_data_file_list(tpc_type="TPC-DS")
    #
    for i in range(len(file_names)):
        fl.load_data(data_generated_dir + "/TPC-DS/" + file_names[i], table_names[i], db_conn)
    #
if tpce_loading_bool == 'True':
    raise NotImplementedError("This logic is not yet implemented!")
#
logger.log("Script Complete!\n-------------------------------------")
