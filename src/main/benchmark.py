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
#
# Prepare database for flashback
db_conn.execute_script(user=ev_loader.var_get('user'),
                       password=ev_loader.var_get('password'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_tearup.sql",
                       params=None)
#
# Revert database post flashback - back to normal state (noarchive mode)
db_conn.execute_script(user=ev_loader.var_get('user'),
                       password=ev_loader.var_get('password'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/flashback_teardown.sql",
                       params=None)
#
# Check whether schema needs creating - executed only if relevant tables are not found
sql_statement = "select count(*) from user_tables where table_name = 'DBGEN_VERSION'"
result = int(db_conn.execute_query(sql_statement, fetch_single=True)[0])
if result == 0:
    db_conn.close()
    raise Exception('[' + ev_loader.var_get('user') + '] schema tables were not found..terminating script!')
#
# Create metric table
xp.create_REP_EXECUTION_PLANS(db_conn=db_conn)
xp.create_REP_EXPLAIN_PLANS(db_conn=db_conn)
#
# Strip optimizer stats
logger.log('Starting optimizer stats dropping..')
OptimizerStatistics.remove_optimizer_statistics(db_conn=db_conn,
                                                logger=logger,
                                                tpctype=ev_loader.var_get('user'))
logger.log('Schema [' + ev_loader.var_get('user') + '] stripped of optimizer stats..')
#
# Start sniffer procedure to terminate long running queries
db_conn.execute_script(user=ev_loader.var_get('user'),
                       password=ev_loader.var_get('password'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/kill_long_running_jobs.sql",
                       params=[ev_loader.var_get('time_out_in_seconds')])
logger.log('Started "kill_long_running" proc')
#
db_conn.close()
#
query_path = ev_loader.var_get("src_dir") + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/Query/"
dml_path = ev_loader.var_get("src_dir") + "/sql/Runtime/TPC-DS/" + ev_loader.var_get('user') + "/DML/"
#
# Execute Queries + DML for n number of iterations
for i in range(1, ev_loader.var_get('iterations') + 1):
    #
    # Execute All Queries
    for j in range(1, 100):
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
    for j in range(1, 43):
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
    db_conn.close()
    logger.log("Executed iteration [" + str(i) + "] of removed stats benchmark")
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start - With Optimizer Stats
------------------------------------------------------------
"""
db_conn.connect()
db_conn.execute_dml(dml='update MON_KILL_LONG_RUNNING set running=0') # Kill Sniffer Procedure
db_conn.commit()
time.sleep(5)
#
# Gather optimizer stats
logger.log('Starting optimizer stats generation..')
OptimizerStatistics.generate_optimizer_statistics(db_conn=db_conn,
                                                  logger=logger,
                                                  tpctype=ev_loader.var_get('user'))
logger.log('Schema [' + ev_loader.var_get('user') + '] has had stats gathered..')
#
db_conn.execute_script(user=ev_loader.var_get('user'),
                       password=ev_loader.var_get('password'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/kill_long_running_jobs.sql",
                       params=[ev_loader.var_get('time_out_in_seconds')])
logger.log('Started "kill_long_running" proc')
db_conn.close()
#
# Execute Queries + DML for n number of iterations
for i in range(1, ev_loader.var_get('iterations')+1):
    #
    # Execute All Queries
    for j in range(1, 100):
        filename = 'query_' + str(j) + '.sql'
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
                                                 gathered_stats=True,
                                                 db_conn=db_conn)
                        xp.generateExplainPlan(sql=sql,
                                               binds=None,
                                               selection=None,
                                               transaction_name=filename,
                                               iteration_run=i,
                                               gathered_stats=True,
                                               db_conn=db_conn)
                        db_conn.close()
    #
    # Execute All DML
    for j in range(1, 43):
        filename = 'dml_' + str(j) + '.sql'
        logger.log('Generating execution metrics for [' + filename + ']..')
        #
        with open(dml_path + filename) as file:
            data = file.read()
            check_if_plsql = XPlan.check_if_plsql_block(statement=data)
            #
            if check_if_plsql:
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
                                             gathered_stats=True,
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
                                                     gathered_stats=True,
                                                     db_conn=db_conn)
                            xp.generateExplainPlan(sql=dml,
                                                   binds=None,
                                                   selection=None,
                                                   transaction_name=filename,
                                                   iteration_run=i,
                                                   gathered_stats=True,
                                                   db_conn=db_conn)
                            db_conn.close()
    db_conn.close()
    logger.log("Executed iteration [" + str(i) + "] of gathered stats benchmark")
"""
SCRIPT CLOSEUP - Cleanup
"""
db_conn.connect()
db_conn.execute_dml(dml='update MON_KILL_LONG_RUNNING set running=0') # Kill Sniffer Procedure
db_conn.commit()
db_conn.close()
# si.initialize_spark().kill_spark_nodes()
logger.log("Script Complete!\n-------------------------------------")