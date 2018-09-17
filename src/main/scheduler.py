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
active_thread_count = 0
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
    [rep_hist_csv.writerow(row) for row in col_list]
    rep_hist_snapshot.close()
    #
    # Create file headers - REP_VSQL_PLAN
    rep_vsql_plan = open(rep_sql_plan_path, 'a')
    rep_plan_csv = csv.writer(rep_vsql_plan, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_vsql_plan',ev_loader=ev_loader,logger=logger)
    [rep_plan_csv.writerow(row) for row in col_list]
    rep_vsql_plan.close()
    #
    # Create file headers - REP_HIST_SYSMETRIC_SUMMARY
    rep_hist_sysmetric_summary = open(rep_hist_sysmetric_summary_path, 'a')
    rep_hist_sysmetric_summary_csv = csv.writer(rep_hist_sysmetric_summary, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_hist_sysmetric_summary', ev_loader=ev_loader, logger=logger)
    [rep_hist_sysmetric_summary_csv.writerow(row) for row in col_list]
    rep_hist_sysmetric_summary.close()
    #
    # Create file headers - REP_HIST_SYSSTAT
    rep_hist_sysstat = open(rep_hist_sysstat_path, 'a')
    rep_hist_sysstat_csv = csv.writer(rep_hist_sysstat, dialect='excel')
    col_list = Workload.get_script_headers(report_type='rep_hist_sysstat', ev_loader=ev_loader, logger=logger)
    [rep_hist_sysstat_csv.writerow(row) for row in col_list]
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
transaction_bank = [] # Keeps reference of which Query/DML scripts are eligible for execution
for j in range(1, 100):
    filename = 'query_' + str(j) + '.sql'
    transaction_bank.append(filename)
for j in range(1, 43):
    filename = 'dml_' + str(j) + '.sql'
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
                                        logger=logger,
                                        path_bank=[rep_hist_snapshot_path,
                                                   rep_sql_plan_path,
                                                   rep_hist_sysmetric_summary_path,
                                                   rep_hist_sysstat_path])
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
        # Pause N seconds between every execution to avoid overwhelming the scheduler
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