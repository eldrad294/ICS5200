"""
--------------------------
This script is used to generate and load TPC data into TPC (DS or E) schema
The script will:
* Generate TPC-E/TPC-DS dat files , according to specified size. A file will be generated per table withing schema
* Generate TPC-E/TPC-DS schema, mainly the tables
* Load each dat file in memory using Spark RDDs.
* The loaded data is dumped into an Oracle instance
* Create indexes on oracle instance
* Generate TPC-E/TPC-DS SQL/DML
--------------------------
"""
"""
---------------------------------------------------
SCRIPT WARM UP - Module Import & Path Configuration
---------------------------------------------------
"""
#
# Module Imports
import sys, threading
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
from src.framework.script_initializer import ScriptInitializer
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir)
ev_loader = si.get_global_config()
db_conn = si.initialize_database()
db_conn.connect()
spark_context = si.initialize_spark().get_spark_context()
logger = si.initialize_logger()

from src.data.tpc import TPC_Wrapper, FileLoader
#
# TPC Wrapper Initialization
tpc = TPC_Wrapper(ev_loader=ev_loader,
                  logger=logger,
                  database_context=db_conn)
"""
----------------------------------
SCRIPT EXECUTION - Data Generation
----------------------------------
"""
if ev_loader.var_get('tpcds_generation_bool') == 'True':
    tpc.generate_data(tpc_type='TPC-DS')
if ev_loader.var_get('tpce_generation_bool') == 'True':
    raise NotImplementedError("This logic is not yet implemented!")
"""
-------------------------------
SCRIPT EXECUTION - Data Loading
-------------------------------
"""
fl = FileLoader(ev_loader=ev_loader,
                logger=logger,
                spark_context=spark_context)
if ev_loader.var_get('tpcds_data_loading_bool') == 'True':
    #
    # Check whether schema needs creating - executed only if relevant tables are not found
    sql_statement = "select count(*) from user_tables where table_name = 'DBGEN_VERSION'"
    result = int(db_conn.execute_query(sql_statement, fetch_single=True)[0])
    if result < 1:
        db_conn.executeScriptsFromFile(ev_loader.var_get("src_dir") + "/sql/Installation/tpcds_schema_tables.sql")
        logger.log('TPC-DS table generation successful!')
    else:
        logger.log('Skipping schema creation..TPC-DS tables already exist!')
    #
    # Retrieve eligible data file names
    table_names = tpc.get_file_extension_list(tpc_type="TPC-DS")[0]
    #
    # Retrieve all eligible data files
    file_names = tpc.get_data_file_list(tpc_type="TPC-DS")
    #
    #thread_pool = []
    for i in range(len(file_names)):
        # t = threading.Thread(target=fl.load_data,
        #                      args=(ev_loader.var_get('data_generated_directory') + "/TPC-DS/" +
        #                            ev_loader.var_get('user') + "/" + file_names[i], table_names[i]),
        #                      name="thread_"+table_names[i])
        # t.start()
        # thread_pool.append(t)
        fl.load_data(path=ev_loader.var_get('data_generated_directory') + "/TPC-DS/" + ev_loader.var_get('user') + "/" + file_names[i],
                     table_name=table_names[i])
    # Barrier waiting for all threads to synchronize
    # for t in thread_pool:
    #     t.join()
    #
    # Check whether indexes needs creating - executed only if relevant indexes are not found
    sql_statement = "select count(*) from user_indexes where index_name = 'SS_SOLD_DATE_SK_INDEX'"
    result = int(db_conn.execute_query(sql_statement, fetch_single=True)[0])
    if result < 1:
        logger.log('Starting schema indexes creation..')
        db_conn.executeScriptsFromFile(ev_loader.var_get("src_dir") + "/sql/Installation/tpcds_schema_indexes.sql")
        logger.log('TPC-DS indexes generation successful!')
    else:
        logger.log('Skipping schema creation..TPC-DS indexes already exist!')
if ev_loader.var_get('tpce_data_loading_bool') == 'True':
    raise NotImplementedError("This logic is not yet implemented!")
"""
------------------------------
SCRIPT EXECUTION - SQL Loading
------------------------------
"""
if ev_loader.var_get('tpcds_sql_generation_bool') == 'True':
    tpc.generate_sql(tpc_type='TPC-DS')
    tpc.split_tpc_sql_file(tpc_type='TPC-DS')
if ev_loader.var_get('tpce_sql_generation_bool') == 'True':
    raise NotImplementedError("This logic is not yet implemented!")
#
logger.log("Script Complete!\n-------------------------------------")
