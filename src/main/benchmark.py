"""
--------------------------
This script is used to execute all TPC provided queries and benchmark them accordingly. The script behaves as follows:
1) Drop all schema optimizer statistics on TPCDSX schema
2) Execute all TPC Queries generated for TPCDSX. Each query execution plan is extracted and returned/saved to disk
   inside table REP_EXECUTION_PLANS
3) Execute all TPC DML generated for TPCDSX. Each dml execution plan is extracted and returned/saved to disk
   inside table REP_EXECUTION_PLANS
4) Repeat Step 2 and 3 for n iterations, as established from config
5) Generate schema wide optimizer statistics for TPCDSX
6) Execute all TPC Queries generated for TPCDSX. Each query execution plan is extracted and returned/saved to disk
   inside table REP_EXECUTION_PLANS
7) Execute all TPC DML generated for TPCDSX. Each dml execution plan is extracted and returned/saved to disk
   inside table REP_EXECUTION_PLANS
8) Repeat Step 6 and 7 for n iterations, as established from config
--------------------------
NB: ENSURE FOLLOWING CONFIG IS ESTABLISHED AND PROPERLY CONFIGURED src/main/config.ini:
1) DatabaseConnectionString.user
2) Benchmark.iterations
--------------------------
"""
"""
---------------------------------------------------
SCRIPT WARM UP - Module Import & Path Configuration
---------------------------------------------------
"""
#
# Module Imports
import sys, os, csv
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
db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                            user=ev_loader.var_get('user'),
                            host=ev_loader.var_get('host'),
                            service=ev_loader.var_get('service'),
                            port=ev_loader.var_get('port'),
                            password=ev_loader.var_get('password'),
                            logger=logger)
xp = XPlan(logger=logger,
           ev_loader=ev_loader)
#
db_conn.connect()
#
# Create metric table
xp.create_REP_EXECUTION_PLANS(db_conn=db_conn)
xp.create_REP_EXPLAIN_PLANS(db_conn=db_conn)
#
csv_rep_execution_plans = "/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_execution_plans.csv"
csv_rep_explain_plans = "/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_explain_plans.csv"
try:
    os.remove(csv_rep_execution_plans)
    os.remove(csv_rep_explain_plans)
except Exception as e:
    logger.log(str(e))
finally:
    rep_execution_plans_file = open(csv_rep_execution_plans, "a")
    rep_explain_plans_file = open(csv_rep_explain_plans, "a")
    #
    execution_output = csv.writer(rep_execution_plans_file, dialect='excel')
    explain_output = csv.writer(rep_explain_plans_file, dialect='excel')
    #
    # Write File Headers
    cur_res, headers = db_conn.execute_query(query='select * from rep_execution_plans', describe=True)
    execution_output.writerow(headers)
    cur_res, headers = db_conn.execute_query(query='select * from rep_explain_plans', describe=True)
    explain_output.writerow(headers)
#
# Check whether schema needs creating - executed only if relevant tables are not found
sql_statement = "select count(*) from user_tables where table_name = 'DBGEN_VERSION'"
result = int(db_conn.execute_query(sql_statement, fetch_single=True)[0])
if result == 0:
    db_conn.close()
    raise Exception('[' + ev_loader.var_get('user') + '] schema tables were not found..terminating script!')
#
# Prepare database for flashback
restore_point_name = ev_loader.var_get('user') + "_benchmark_rp"
db_conn.execute_script(user=ev_loader.var_get('sysuser'),
                       password=ev_loader.var_get('syspassword'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_tearup.sql",
                       params=[restore_point_name])
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start
------------------------------------------------------------
"""
#
query_path = ev_loader.var_get("src_dir") + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Query/"
dml_path = ev_loader.var_get("src_dir") + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/DML/"
#
# Execute Queries + DML for n number of iterations
for i in range(1, (ev_loader.var_get('iterations') + 1) * 2):
    #
    # Database connection would have to be reopened at this point, due to db restart
    db_conn.connect()
    #
    # Drop stats during first half of the benchmark, Gather stats during first half of the benchmark.
    if i > (ev_loader.var_get('iterations')):
        #
        # Gather optimizer stats
        logger.log('Starting optimizer stats generation..')
        OptimizerStatistics.generate_optimizer_statistics(db_conn=db_conn,
                                                          logger=logger,
                                                          tpctype=ev_loader.var_get('user'))
        logger.log('Schema [' + ev_loader.var_get('user') + '] has had stats gathered..')
    else:
        #
        # Strip optimizer stats
        logger.log('Starting optimizer stats dropping..')
        OptimizerStatistics.remove_optimizer_statistics(db_conn=db_conn,
                                                        logger=logger,
                                                        tpctype=ev_loader.var_get('user'))
        logger.log('Schema [' + ev_loader.var_get('user') + '] stripped of optimizer stats..')
    #
    db_conn.close()
    #
    # Execute All Queries
    for j in range(99, 100):
        filename = 'query_'+str(j)+'.sql'
        with open(query_path + filename) as file:
            logger.log('Generating execution metrics for [' + filename + ']..')
            data = file.read()
            sql_list = data.split(';')
            for sql in sql_list:
                sql = sql.replace("\n", " ")
                if sql.isspace() is not True and sql != "":
                    sql = xp.execution_plan_syntax(sql)
                    try:
                        db_conn.connect()
                        db_conn.execute_dml(dml=sql, params=None)
                        logger.log('Successfully executed [' + filename + "]")
                    except Exception as e:
                        logger.log(str(e))
                    finally:
                        db_conn.close()
                        db_conn.connect()
                        xp.generateExecutionPlan(sql=sql,
                                                 binds=None,
                                                 selection=None,
                                                 transaction_name=filename,
                                                 iteration_run=i,
                                                 gathered_stats=False,
                                                 db_conn=db_conn)
                        xp.generateExplainPlan(sql=sql,
                                               binds=None,
                                               selection=None,
                                               transaction_name=filename,
                                               iteration_run=i,
                                               gathered_stats=False,
                                               db_conn=db_conn)
                        db_conn.close()
    #
    # Execute All DML
    for j in range(42, 43):
        filename = 'dml_' + str(j) + '.sql'
        logger.log('Generating execution metrics for [' + filename + ']..')
        #
        with open(dml_path + filename) as file:
            data = file.read()
            check_if_plsql = XPlan.check_if_plsql_block(statement=data)
            #
            if check_if_plsql:
                #
                # Executes PL/SQL block
                sql = xp.execution_plan_syntax(data)
                try:
                    db_conn.connect()
                    db_conn.execute_dml(dml=sql, params=None)
                    logger.log('Successfully executed [' + filename + "]")
                except Exception as e:
                    logger.log(str(e))
                finally:
                    db_conn.close()
                    db_conn.connect()
                    xp.generateExecutionPlan(sql=sql,
                                             binds=None,
                                             selection=None,
                                             transaction_name=filename,
                                             iteration_run=i,
                                             gathered_stats=False,
                                             db_conn=db_conn)
                    db_conn.close()
            else:
                # Executes statements as a series of sql statements
                dml_list = data.split(';')
                for dml in dml_list:
                    dml = dml.replace("\n"," ")
                    if dml.isspace() is not True and dml != "":
                        dml = xp.execution_plan_syntax(dml)
                        try:
                            db_conn.connect()
                            db_conn.execute_dml(dml=dml, params=None)
                            logger.log('Successfully executed [' + filename + "]")
                        except Exception as e:
                            logger.log(str(e))
                        finally:
                            db_conn.close()
                            db_conn.connect()
                            xp.generateExecutionPlan(sql=dml,
                                                     binds=None,
                                                     selection=None,
                                                     transaction_name=filename,
                                                     iteration_run=i,
                                                     gathered_stats=False,
                                                     db_conn=db_conn)
                            xp.generateExplainPlan(sql=dml,
                                                   binds=None,
                                                   selection=None,
                                                   transaction_name=filename,
                                                   iteration_run=i,
                                                   gathered_stats=False,
                                                   db_conn=db_conn)
                            db_conn.close()
    #
    # Offload rep_execution_plans & rep_explain_plans into csv
    #
    db_conn.connect()
    cur_res = db_conn.execute_query(query='select * from rep_execution_plans')
    [execution_output.writerow(row) for row in cur_res]
    cur_res = db_conn.execute_query(query='select * from rep_explain_plans')
    [explain_output.writerow(row) for row in cur_res]
    db_conn.close()
    #
    # Enable Flashback
    db_conn.execute_script(user=ev_loader.var_get('sysuser'),
                           password=ev_loader.var_get('syspassword'),
                           instance_name=ev_loader.var_get('instance_name'),
                           filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_start.sql",
                           params=[restore_point_name])
    #
    if i > (ev_loader.var_get('iterations')):
        logger.log("Executed iteration [" + str(i) + "] of gathered stats benchmark")
    else:
        logger.log("Executed iteration [" + str(i) + "] of removed stats benchmark")
"""
SCRIPT CLOSEUP - Cleanup
"""
#
# Close CSV file
rep_execution_plans_file.close()
rep_explain_plans_file.close()
#
# Revert database post flashback - back to normal state (noarchive mode)
db_conn.execute_script(user=ev_loader.var_get('sysuser'),
                       password=ev_loader.var_get('syspassword'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_teardown.sql",
                       params=[restore_point_name])
# si.initialize_spark().kill_spark_nodes()
logger.log("Script Complete!\n-------------------------------------")