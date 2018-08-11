<<<<<<< HEAD
from src.reports.bar import BarCharts
from src.framework.logger import Logger
from src.framework.db_interface import ConnectionPool
connection_details = {'instance_name':'gabsam',
                              'user':'tpcds1',
                              'host':'192.168.202.222',
                              'service':'gabsam',
                              'port':'1521',
                              'password':'tpc'}
logger = Logger(log_file_path=None,
                     write_to_disk='false',
                     write_to_screen='true')
ConnectionPool.create_connection_pool(max_connections=1,
                                      connection_details=connection_details,
                                      logger=logger)
=======
"""
--------------------------
This script is used to generate reports
The script will:
* TBD
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
>>>>>>> faca480c8098f2b884dd86e3843e6d6beca3b0a3
#
"""
----------------------------------
SCRIPT EXECUTION - Report Generation
----------------------------------
"""
bc = BarCharts(ConnectionPool.claim_from_pool())
bc.generate_REP_TPC_DESCRIBE()
"""
SCRIPT CLOSEUP - Cleanup
"""
ConnectionPool.close_connection_pool()
# si.initialize_spark().kill_spark_nodes()
logger.log("Script Complete!\n-------------------------------------")