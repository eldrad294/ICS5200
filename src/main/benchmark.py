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
#
from src.framework.script_initializer import ScriptInitializer
from src.framework.db_interface import DatabaseInterface
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir)
ev_loader = si.get_global_config()
logger = si.initialize_logger()
from src.utils.plan_control import XPlan
from src.utils.stats_control import OptimizerStatistics
from src.utils.flashback_control import FlashbackControl
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start - Without Optimizer Stats
------------------------------------------------------------
"""
db_conn_info = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                            user=ev_loader.var_get('user'),
                            host=ev_loader.var_get('host'),
                            service=ev_loader.var_get('service'),
                            port=ev_loader.var_get('port'),
                            password=ev_loader.var_get('password'),
                            logger=logger)
db_conn2 = db_conn_info
db_conn2.connect()
xp = XPlan(db_conn=db_conn2,
           logger=logger,
           ev_loader=ev_loader)
#
# Check whether schema needs creating - executed only if relevant tables are not found
db_conn = db_conn_info
db_conn.connect()
sql_statement = "select count(*) from user_tables where table_name = 'DBGEN_VERSION'"
result = int(db_conn.execute_query(sql_statement, fetch_single=True)[0])
if result == 0:
    raise Exception('[' + ev_loader.var_get('user') + '] schema tables were not found..terminating script!')
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
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/run_kill_long_running_jobs.sql",
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
    # #
    # Execute All Queries
    # for j in range(1, 100):
    #     filename = 'query_'+str(j)+'.sql'
    #     with open(query_path + filename) as file:
    #         logger.log('Generating execution metrics for [' + filename + ']..')
    #         data = file.read()
    #         sql_list = data.split(';')
    #         for sql in sql_list:
    #             sql = sql.replace("\n", " ")
    #             if sql.isspace() is not True and sql != "":
    #                 sql = xp.execution_plan_syntax(sql)
    #                 try:
    #                     db_conn = db_conn_info
    #                     db_conn.connect()
    #                     db_conn.execute_dml(dml=sql, params=None)
    #                 except Exception as e:
    #                     logger.log(str(e))
    #                 finally:
    #                     db_conn.close()
    #                     db_conn = db_conn_info
    #                     db_conn.connect()
    #                     xp.generateExecutionPlan(sql=sql,
    #                                              binds=None,
    #                                              selection=None,
    #                                              transaction_name=filename,
    #                                              iteration_run=i,
    #                                              gathered_stats=False,
    #                                              db_conn=db_conn)
    #                     db_conn.close()
    # Execute All DML
    for j in range(1, 43):
        filename = 'dml_' + str(j) + '.sql'
        logger.log('Generating execution metrics for [' + filename + ']..')
        #
        # Keep reference to flashback timestamp
        ts = FlashbackControl.captureTimeStamp()
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
                    db_conn = db_conn_info
                    db_conn.connect()
                    db_conn.execute_dml(dml=sql, params=None)
                except Exception as e:
                    logger.log(str(e))
                finally:
                    db_conn.close()
                    db_conn = db_conn_info
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
                            db_conn = db_conn_info
                            db_conn.connect()
                            db_conn.execute_dml(dml=dml, params=None)
                        except Exception as e:
                            logger.log(str(e))
                        finally:
                            db_conn.close()
                            db_conn = db_conn_info
                            db_conn.connect()
                            xp.generateExecutionPlan(sql=dml,
                                                     binds=None,
                                                     selection=None,
                                                     transaction_name=filename,
                                                     iteration_run=i,
                                                     gathered_stats=False,
                                                     db_conn=db_conn)
                            db_conn.close()
        #
        # Flashback Impacted Tables
        db_conn = db_conn_info
        db_conn.connect()
        FlashbackControl.flashback_tables(db_conn=db_conn,
                                          logger=logger,
                                          timestamp=ts,
                                          ev_loader=ev_loader)
        db_conn.close()
    logger.log("Executed iteration [" + str(i) + "] of removed stats benchmark")
"""
------------------------------------------------------------
SCRIPT EXECUTION - Benchmark Start - With Optimizer Stats
------------------------------------------------------------
"""
#
# Gather optimizer stats
db_conn = db_conn_info
db_conn.connect()
db_conn.execute_dml(dml='update MON_KILL_LONG_RUNNING set running=0') # Kill Sniffer Procedure
db_conn.commit()
time.sleep(5)
logger.log('Starting optimizer stats generation..')
OptimizerStatistics.generate_optimizer_statistics(db_conn=db_conn,
                                                  logger=logger,
                                                  tpctype=ev_loader.var_get('user'))
logger.log('Schema [' + ev_loader.var_get('user') + '] stripped of optimizer stats..')
#
db_conn.execute_script(user=ev_loader.var_get('user'),
                       password=ev_loader.var_get('password'),
                       instance_name=ev_loader.var_get('instance_name'),
                       filename=ev_loader.var_get("src_dir") + "/sql/Utility/run_kill_long_running_jobs.sql",
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
                        db_conn = db_conn_info
                        db_conn.connect()
                        db_conn.execute_dml(dml=sql, params=None)
                    except Exception as e:
                        logger.log(str(e))
                    finally:
                        db_conn.close()
                        db_conn = db_conn_info
                        db_conn.connect()
                        xp.generateExecutionPlan(sql=sql,
                                                 binds=None,
                                                 selection=None,
                                                 transaction_name=filename,
                                                 iteration_run=i,
                                                 gathered_stats=False,
                                                 db_conn=db_conn)
                        db_conn.close()
    # Execute All DML
    for j in range(1, 43):
        filename = 'dml_' + str(j) + '.sql'
        logger.log('Generating execution metrics for [' + filename + ']..')
        #
        # Keep reference to flashback timestamp
        ts = FlashbackControl.captureTimeStamp()
        with open(dml_path + filename) as file:
            data = file.read()
            check_if_plsql = XPlan.check_if_plsql_block(statement=data)
            #
            if check_if_plsql:
                sql = xp.execution_plan_syntax(data)
                try:
                    db_conn = db_conn_info
                    db_conn.connect()
                    db_conn.execute_dml(dml=sql, params=None)
                except Exception as e:
                    logger.log(str(e))
                finally:
                    db_conn.close()
                    db_conn = db_conn_info
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
                            db_conn = db_conn_info
                            db_conn.connect()
                            db_conn.execute_dml(dml=dml, params=None)
                        except Exception as e:
                            logger.log(str(e))
                        finally:
                            db_conn.close()
                            db_conn = db_conn_info
                            db_conn.connect()
                            xp.generateExecutionPlan(sql=dml,
                                                     binds=None,
                                                     selection=None,
                                                     transaction_name=filename,
                                                     iteration_run=i,
                                                     gathered_stats=False,
                                                     db_conn=db_conn)
                            db_conn.close()
        #
        # Flashback Impacted Tables
        db_conn = db_conn_info
        db_conn.connect()
        FlashbackControl.flashback_tables(db_conn=db_conn,
                                          logger=logger,
                                          timestamp=ts,
                                          ev_loader=ev_loader)
        db_conn.close()
    logger.log("Executed iteration [" + str(i) + "] of gathered stats benchmark")
"""
SCRIPT CLOSEUP - Cleanup
"""
db_conn = db_conn_info
db_conn.connect()
db_conn.execute_dml(dml='update MON_KILL_LONG_RUNNING set running=0') # Kill Sniffer Procedure
db_conn.commit()
db_conn.close()
db_conn2.close()
# si.initialize_spark().kill_spark_nodes()
logger.log("Script Complete!\n-------------------------------------")