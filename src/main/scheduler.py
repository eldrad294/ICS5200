"""
---------------------------------------------------
SCRIPT WARM UP - Module Import & Path Configuration
---------------------------------------------------
"""
#
# Module Imports
import sys, os, time
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
from src.data.workload import Workload
from src.framework.db_interface import DatabaseInterface
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir, log_name_prefix=file_name)
ev_loader = si.get_global_config()
logger = si.initialize_logger()
active_thread_count = 0
"""
------------------------------------------------------------
SCRIPT EXECUTION - Workload Start
------------------------------------------------------------
"""
query_path = ev_loader.var_get('src_dir')+"/sql/Runtime/TPC-DS/" + ev_loader.var_get("user") + "/Query/"
dml_path = ev_loader.var_get('src_dir')+"/sql/Runtime/TPC-DS/" + ev_loader.var_get("user") + "/DML/"
#
transaction_bank = [] # Keeps reference of which Query/DML scripts are eligible for execution
for filename in os.listdir(query_path):
    if filename.endswith(".sql"):
        transaction_bank.append(filename)
for filename in os.listdir(dml_path):
    if filename.endswith(".sql"):
        transaction_bank.append(filename)
#
# Prepare database for flashback
restore_point_name = ev_loader.var_get('user') + "_scheduler_rp"
DatabaseInterface.execute_script(user=ev_loader.var_get('sysuser'),
                                 password=ev_loader.var_get('syspassword'),
                                 instance_name=ev_loader.var_get('instance_name'),
                                 filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_tearup.sql",
                                 params=None,
                                 logger=logger)
#
while True:
    #
    # Create restore point
    DatabaseInterface.execute_script(user=ev_loader.var_get('sysuser'),
                                     password=ev_loader.var_get('syspassword'),
                                     instance_name=ev_loader.var_get('instance_name'),
                                     filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_restore_create.sql",
                                     params=[restore_point_name],
                                     logger=logger)
    logger.log('Created restore point ' + restore_point_name + '..')
    #
    # This thread oversees metric extraction and saves to local generated files
    Workload.execute_statistic_gatherer(ev_loader=ev_loader,
                                        logger=logger)
    #
    for script in transaction_bank:
        #
        if 'query' in script:
            transaction_path = query_path
        elif 'dml' in script:
            transaction_path = dml_path
        else:
            logger.log('Script name malformed!')
            raise LookupError('Script name malformed!')
        #
        # Execute script on a forked process
        Workload.execute_transaction(ev_loader=ev_loader,
                                     logger=logger,
                                     transaction_path=transaction_path,
                                     transaction_name=script)
        #
        # Block further spawning of threads beyond the limit dictated by 'ev_loader.var_get("parallel_cap")'
        Workload.barrier(ev_loader=ev_loader)
        #
        # Pause N seconds between every execution to avoid overwhelming the schedule
        time.sleep(ev_loader.var_get('execution_intervals'))
    #
    # Enable Flashback
    DatabaseInterface.execute_script(user=ev_loader.var_get('sysuser'),
                                     password=ev_loader.var_get('syspassword'),
                                     instance_name=ev_loader.var_get('instance_name'),
                                     filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_start.sql",
                                     params=[restore_point_name],
                                     logger=logger)
    logger.log('Flash backed to ' + restore_point_name + "..")
    #
    # Drop Restore Point
    DatabaseInterface.execute_script(user=ev_loader.var_get('sysuser'),
                                     password=ev_loader.var_get('syspassword'),
                                     instance_name=ev_loader.var_get('instance_name'),
                                     filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_restore_drop.sql",
                                     params=[restore_point_name],
                                     logger=logger)
    logger.log('Dropped restore point ' + restore_point_name + '..')
    #
    # Delete alert / trace files generated by database '/oracle/diag/rdbms/gabsam/gabsam/'
    if ev_loader.var_get('delete_trace_alert_logs') == 'True':
        sys = 'rm /oracle/diag/rdbms/gabsam/gabsam/alert/* /oracle/diag/rdbms/gabsam/gabsam/trace/*'
        output = os.system(sys)
        if output != 0:
            logger.log('Removal of oracle alter/trace file raised error code ' + str(output) + '!')