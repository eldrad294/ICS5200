"""
---------------------------------------------------
SCRIPT WARM UP - Module Import & Path Configuration
---------------------------------------------------
"""
#
# Module Imports
import sys, os, time, csv
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
from src.utils.stats_control import OptimizerStatistics
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir, log_name_prefix=file_name)
ev_loader = si.get_global_config()
logger = si.initialize_logger()
db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                            user=ev_loader.var_get('user'),
                            host=ev_loader.var_get('host'),
                            service=ev_loader.var_get('service'),
                            port=ev_loader.var_get('port'),
                            password=ev_loader.var_get('password'),
                            logger=logger)
#
# Makes relavent checks to ensure metric csv files exist
rep_hist_snapshot_path = ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Schedule/rep_hist_snapshot.csv"
rep_sql_plan_path = ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Schedule/rep_vsql_plan.csv"
rep_hist_sysmetric_summary_path = ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Schedule/rep_hist_sysmetric_summary.csv"
rep_hist_sysstat_path = ev_loader.var_get('src_dir') + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Schedule/rep_hist_sysstat.csv"
rep_hist_snapshot_exists = os.path.isfile(rep_hist_snapshot_path)
rep_sql_plan_exists = os.path.isfile(rep_sql_plan_path)
rep_hist_sysmetric_summary_exists = os.path.isfile(rep_hist_sysmetric_summary_path)
rep_hist_sysstat_exists = os.path.isfile(rep_hist_sysstat_path)
#
if ev_loader.var_get('renew_csv') == 'True':
    #
    if rep_hist_snapshot_exists:
        os.remove(rep_hist_snapshot_path)
    if rep_sql_plan_exists:
        os.remove(rep_sql_plan_path)
    if rep_hist_sysmetric_summary_exists:
        os.remove(rep_hist_sysmetric_summary_path)
    if rep_hist_sysstat_exists:
        os.remove(rep_hist_sysstat_path)
    #
    os.mknod(rep_hist_snapshot_path)
    logger.log('Created file ' + rep_hist_snapshot_path)
    os.mknod(rep_sql_plan_path)
    logger.log('Created file ' + rep_sql_plan_path)
    os.mknod(rep_hist_sysmetric_summary_path)
    logger.log('Created file ' + rep_hist_sysmetric_summary_path)
    os.mknod(rep_hist_sysstat_path)
    logger.log('Created file ' + rep_hist_sysstat_path)
    #
    # Create file headers - REP_HIST_SNAPSHOT
    rep_hist_snapshot = open(rep_hist_snapshot_path, 'a')
    rep_hist_csv = csv.writer(rep_hist_snapshot, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_hist_snapshot',ev_loader=ev_loader,logger=logger)
    rep_hist_csv.writerow(col_list)
    rep_hist_snapshot.close()
    #
    # Create file headers - REP_VSQL_PLAN
    rep_vsql_plan = open(rep_sql_plan_path, 'a')
    rep_plan_csv = csv.writer(rep_vsql_plan, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_vsql_plan',ev_loader=ev_loader,logger=logger)
    rep_plan_csv.writerow(col_list)
    rep_vsql_plan.close()
    #
    # Create file headers - REP_HIST_SYSMETRIC_SUMMARY
    rep_hist_sysmetric_summary = open(rep_hist_sysmetric_summary_path, 'a')
    rep_hist_sysmetric_summary_csv = csv.writer(rep_hist_sysmetric_summary, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_hist_sysmetric_summary', ev_loader=ev_loader, logger=logger)
    rep_hist_sysmetric_summary_csv.writerow(col_list)
    rep_hist_sysmetric_summary.close()
    #
    # Create file headers - REP_HIST_SYSSTAT
    rep_hist_sysstat = open(rep_hist_sysstat_path, 'a')
    rep_hist_sysstat_csv = csv.writer(rep_hist_sysstat, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_hist_sysstat', ev_loader=ev_loader, logger=logger)
    rep_hist_sysstat_csv.writerow(col_list)
    rep_hist_sysstat.close()
    #
elif ev_loader.var_get('renew_csv') == 'False':
    #
    if not rep_hist_snapshot_exists:
        raise FileNotFoundError(rep_hist_snapshot_path)
    if not rep_sql_plan_exists:
        raise FileNotFoundError(rep_sql_plan_path)
    if not rep_hist_sysmetric_summary_exists:
        raise FileNotFoundError(rep_hist_sysmetric_summary_path)
    if not rep_hist_sysstat_exists:
        raise FileNotFoundError(rep_hist_sysstat_path)
"""
------------------------------------------------------------
SCRIPT EXECUTION - Workload Start
------------------------------------------------------------
"""
db_conn.connect()
#
# Gather optimizer stats
logger.log('Starting optimizer stats generation..')
OptimizerStatistics.generate_optimizer_statistics(db_conn=db_conn,
                                                  logger=logger,
                                                  tpctype=ev_loader.var_get('user'))
logger.log('Schema [' + ev_loader.var_get('user') + '] has had stats gathered..')
db_conn.close()
#
query_path = ev_loader.var_get('src_dir')+"/sql/Runtime/TPC-DS/" + ev_loader.var_get("user") + "/Query/"
dml_path = ev_loader.var_get('src_dir')+"/sql/Runtime/TPC-DS/" + ev_loader.var_get("user") + "/DML/"
#
query_bank, dml_bank = [], [] # Keeps reference of which Query/DML scripts are eligible for execution
for j in range(1, 100):
    filename = 'query_' + str(j) + '.sql'
    query_bank.append(filename)
for j in range(1, 43):
    filename = 'dml_' + str(j) + '.sql'
    dml_bank.append(filename)