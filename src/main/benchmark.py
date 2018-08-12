"""
--------------------------
This script is used to execute all TPC provided queries and benchmark them accordingly
--------------------------
"""
"""
---------------------------------------------------
SCRIPT WARM UP - Module Import & Path Configuration
---------------------------------------------------
"""
#
# Module Imports
import sys
from os.path import dirname, abspath
#
# Retrieving relative paths for project directory
home_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
project_dir = dirname(dirname(dirname(abspath(__file__))))
src_dir = dirname(dirname(abspath(__file__)))
#
# Appending to python path
sys.path.append(home_dir)
sys.path.append(project_dir)
sys.path.append(src_dir)
#
from src.framework.script_initializer import ScriptInitializer
from src.framework.db_interface import ConnectionPool
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir)
ev_loader = si.get_global_config()
db_conn = ConnectionPool.claim_from_pool()[2]
spark_context = si.initialize_spark().get_spark_context()
logger = si.initialize_logger()
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start - Without Optimizer Stats
------------------------------------------------------------
"""
#
# Check whether schema needs creating - executed only if relevant tables are not found
sql_statement = "select count(*) from user_tables where table_name = 'DBGEN_VERSION'"
result = int(db_conn.execute_query(sql_statement, fetch_single=True)[0])
if result == 0:
    raise Exception('[' + ev_loader.var_get('user') + '] schema tables were not found..terminating script!')
#
# Strip optimizer stats
db_conn.executeScriptsFromFile(ev_loader.var_get("src_dir") + "/sql/Utility/stats_removal/stats_removal_" + ev_loader.var_get('user') + ".sql")
logger.log('Schema [' + ev_loader.var_get('user') + '] stripped of optimizer stats..')
#
# Execute Queries + DML for n number of iterations
query_path = ev_loader.var_get("src_dir") + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Query/"
dml_path = ev_loader.var_get("src_dir") + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/DML/"
for i in range(5):
    # Execute All Queries
    for filename in os.listdir(query_path):
        db_conn.executeScriptsFromFile(query_path + filename)
    # Execute All DML
    for filename in os.listdir(dml_path):
        db_conn.executeScriptsFromFile(dml_path + filename)
    logger.log("Executed iteration [" + str(i) + "] of removed stats benchmark")
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start - With Optimizer Stats
------------------------------------------------------------
"""
#
# Gather optimizer stats
db_conn.executeScriptsFromFile(ev_loader.var_get("src_dir") + "/sql/Utility/stats_generation/stats_generation_" + ev_loader.var_get('user') + ".sql")
logger.log('Schema [' + ev_loader.var_get('user') + '] stripped of optimizer stats..')
#
# Execute Queries + DML for n number of iterations
for i in range(5):
    # Execute All Queries
    for filename in os.listdir(query_path):
        db_conn.executeScriptsFromFile(query_path + filename)
    # Execute All DML
    for filename in os.listdir(dml_path):
        db_conn.executeScriptsFromFile(dml_path + filename)
    logger.log("Executed iteration [" + str(i) + "] of gathered stats benchmark")