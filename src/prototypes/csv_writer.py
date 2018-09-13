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
from src.framework.db_interface import DatabaseInterface
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir, log_name_prefix=file_name)
ev_loader = si.get_global_config()
logger = si.initialize_logger()
from src.utils.plan_control import XPlan
from src.utils.stats_control import OptimizerStatistics
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start - Without Optimizer Stats
------------------------------------------------------------
"""
import csv
open("/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_execution_plans.csv","w")
open("/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_explain_plans.csv","w")
rep_execution_plans_file=open("/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_execution_plans.csv","a")
rep_explain_plans_file=open("/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_explain_plans.csv","a")
execution_output=csv.writer(rep_execution_plans_file, dialect='csv')
explain_output=csv.writer(rep_explain_plans_file, dialect='csv')
db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                            user=ev_loader.var_get('user'),
                            host=ev_loader.var_get('host'),
                            service=ev_loader.var_get('service'),
                            port=ev_loader.var_get('port'),
                            password=ev_loader.var_get('password'),
                            logger=logger)
xp = XPlan(logger=logger,
           ev_loader=ev_loader)
db_conn.connect()
cur_res = db_conn.execute_query(query='select * from rep_execution_plans')
for row in cur_res:
    execution_output.writerow(row)
cur_res = db_conn.execute_query(query='select * from rep_explain_plans')
for row in cur_res:
    explain_output.writerow(row)
db_conn.close()