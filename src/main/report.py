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
import sys, os
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
file_name = os.path.basename(__file__).split('.')[0]
#
from src.framework.script_initializer import ScriptInitializer
from src.framework.db_interface import ConnectionPool
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir, log_name_prefix=file_name)
ev_loader = si.get_global_config()
db_conn = ConnectionPool.claim_from_pool()[2]
#spark_context = si.initialize_spark().get_spark_context()
logger = si.initialize_logger()
#
"""
------------------------------------
SCRIPT EXECUTION - Report Generation
------------------------------------
"""
from src.reports.bar import BarCharts
bc = BarCharts(db_conn, logger, ev_loader.var_get('report_save_path'))
#
# This line requires SQL/Reports/REP_TPC_DESCRIBE.sql to have been executed for respective schema
# bc.generate_REP_TPC_DESCRIBE(tpc_type=ev_loader.var_get('user'))
columns = ['ELAPSED_TIME_MINS',
          'SORTS',
          'PARSE_CALLS',
          'DISK_READS',
          'DIRECT_WRITES',
          'BUFFER_GETS',
          'USER_IO_WAIT_TIME',
          'OPTIMIZER_COST',
          'CPU_TIME_MINS',
          'IO_INTERCONNECT_BYTES',
          'PHYSICAL_READ_REQUESTS',
          'PHYSICAL_WRITE_REQUESTS',
          'SHARABLE_MEM',
          'PERSISTENT_MEM',
          'RUNTIME_MEM',
          'PLSQL_EXEC_TIME_MINS']
bc.generate_REP_EXECUTION_PLANS(ev_loader=ev_loader,gathered_stats=False,iterations=3,columns=columns,from_table=False)
bc.generate_REP_EXECUTION_PLANS(ev_loader=ev_loader,gathered_stats=True,iterations=3,columns=columns,from_table=False)
"""
SCRIPT CLOSEUP - Cleanup
"""
ConnectionPool.close_connection_pool()
# si.initialize_spark().kill_spark_nodes()
logger.log("Script Complete!\n-------------------------------------")