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
if ev_loader.var_get('enable_spark') == 'True':
    spark_context = si.initialize_spark().get_spark_context()
else:
    spark_context = None
#
# TPC Wrapper Initialization
tpc = TPC_Wrapper(ev_loader=ev_loader,
                  logger=logger,
                  database_context=db_conn)
#
# File Loading Initialization
fl = FileLoader(ev_loader=ev_loader,
                logger=logger,
                spark_context=spark_context)
#
# Makes relevent checks to ensure metric csv files exist
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
#
def __load_and_delete(tpc):
    """
    Wrapper method which contains functionality used to invoke sqlldr, load data, and delete the TPC-DS generated files
    :param tpc: Instance of class 'tpc.py'
    :return: None
    """
    #
    # Retrieve eligible data file names
    table_names = tpc.get_file_extension_list(tpc_type="TPC-DS")[0]
    #
    # Retrieve all eligible data files
    file_names = tpc.get_data_file_list(tpc_type="TPC-DS")
    #
    for i in range(len(file_names)):
        #
        # Loads data into Oracle Instance using Spark
        # fl.load_data(path=ev_loader.var_get('data_generated_directory') + "/TPC-DS/" + ev_loader.var_get('user') + "/" + file_names[i],
        #              table_name=table_names[i])
        #
        # Loads data through SQL Loader control files
        fl.call_ctrl_file(tpcds_type="tpcds", table_name=table_names[i])
        #
        # Deletes generated data file
        if ev_loader.var_get('data_retain_bool') == 'False':
            tpc.delete_data(tpc_type="TPC-DS", file_name=file_names[i])
#
def __power_test(tpc, ev_loader, logger):
    #
    # Retrieve query stream sequence
    query_stream = tpc.get_order_sequence(stream_identification_number=0, tpc_type='TPC-DS')
    #
    for i in range(0, len(query_stream)):
        query_name = 'query_' + str(query_stream[i]) + '.sql'
        #
        DatabaseInterface.execute_script(user=ev_loader.var_get('user'),
                                         password=ev_loader.var_get('password'),
                                         instance_name=ev_loader.var_get('instance_name'),
                                         filename=query_path + query_name,
                                         params=None,
                                         logger=logger,
                                         redirect_path=ev_loader.var_get('project_dir') + "/log/sqlplusoutput.txt")
#
def __throughput_test(tpc, ev_loader, logger):
    #
    # Iterate over all query streams and execute in parallel
    for i in range(0, ev_loader.var_get('stream_total')+1):
        #
        # Retrieve query stream sequence
        query_stream = tpc.get_order_sequence(stream_identification_number=i, tpc_type='TPC-DS')
        #
        # Execute script on a forked process
        Workload.execute_transaction(ev_loader=ev_loader,
                                     logger=logger,
                                     transaction_path=transaction_path,
                                     query_stream=query_stream)
    #
    # Create Barrier to allow all parallel executions to finish

#
"""
------------------------------------------------------------
SCRIPT EXECUTION - Workload Loop
------------------------------------------------------------
"""
#
# This thread oversees metric extraction and saves to local generated files
Workload.execute_statistic_gatherer(ev_loader=ev_loader,
                                    logger=logger,
                                    path_bank=[rep_hist_snapshot_path,
                                               rep_sql_plan_path,
                                               rep_hist_sysmetric_summary_path,
                                               rep_hist_sysstat_path])
while True:
    #
    # 1) Dropping prior schema
    logger.log('Dropping schema [' + ev_loader.var_get('user') + ']..')
    DatabaseInterface.execute_script(user=ev_loader.var_get('user'),
                                     password=ev_loader.var_get('password'),
                                     instance_name=ev_loader.var_get('instance_name'),
                                     filename=ev_loader.var_get("src_dir") + "/sql/Rollback/rb_tpcds_schema.sql",
                                     params=None,
                                     logger=logger)
    #
    # 2) Creating new schema for preparation of data loading
    logger.log('Preparing schema [' + ev_loader.var_get('user') + ']..')
    DatabaseInterface.execute_script(user=ev_loader.var_get('user'),
                                     password=ev_loader.var_get('password'),
                                     instance_name=ev_loader.var_get('instance_name'),
                                     filename=ev_loader.var_get("src_dir") + "/sql/Installation/schema_tables_" + ev_loader.var_get('user') + ".sql",
                                     params=None,
                                     logger=logger)
    #
    # 3.1) Creating TPC-DS Data
    logger.log('Generating data of ' + str(ev_loader.var_get('data_size')) + 'G..')
    tpc.generate_data(tpc_type='TPC-DS')
    #
    # 3.2) Loading TPC-DS Data
    logger.log('Loading data into schema [' + ev_loader.var_get('user') + ']')
    __load_and_delete(tpc=tpc)
    #
    # 4) Creating indexes on loaded data
    logger.log('Building indexes on schema [' + ev_loader.var_get('user') + ']')
    DatabaseInterface.execute_script(user=ev_loader.var_get('user'),
                                     password=ev_loader.var_get('password'),
                                     instance_name=ev_loader.var_get('instance_name'),
                                     filename=ev_loader.var_get("src_dir") + "/sql/Installation/schema_indexes_" + ev_loader.var_get('user') + ".sql",
                                     params=None,
                                     logger=logger)
    #
    # 5) Gather Optimizer Statistics
    logger.log('Gathering database wide optimizer statistics for schema [' + ev_loader.var_get('user') + ']')
    db_conn.connect()
    OptimizerStatistics.generate_optimizer_statistics(db_conn=db_conn,
                                                      logger=logger,
                                                      tpctype=ev_loader.var_get('user'))
    db_conn.close()
    #
    # 6) Power Test
    logger.log("Initiating power test for schema [" + ev_loader.var_get('user') + "]..")
    __power_test(tpc=tpc, ev_loader=ev_loader, logger=logger)
    #
    # 7) Throughput Test 1
    logger.log("Initiating throughput test 1 for schema [" + ev_loader.var_get('user') + "]..")
    __throughput_test(tpc=tpc, ev_loader=ev_loader, logger=logger)