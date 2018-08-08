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

from src.data.tpc import TPC_Wrapper, FileLoader
from src.framework.db_interface import DatabaseInterface
#
# TPC Wrapper Initialization
tpc = TPC_Wrapper(ev_loader=ev_loader,
                  logger=logger,
                  database_context=db_conn)
#
db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                                 user=ev_loader.var_get('user'),
                                 host=ev_loader.var_get('host'),
                                 service=ev_loader.var_get('service'),
                                 port=ev_loader.var_get('port'),
                                 password=ev_loader.var_get('password'),
                                 logger=logger)
db_conn.connect()
sql = "select column from user_tab_columns where table_name = 'INVENTORY'";
res = db_conn.execute_query(query=sql, describe=False)
for item in res:
    print(item[0])