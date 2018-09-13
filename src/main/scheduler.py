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
while True:
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
        time.sleep(ev_loader.var_get('intervals'))
#