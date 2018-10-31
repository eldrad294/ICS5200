#
# Module Imports
from src.framework.db_interface import DatabaseInterface
from src.utils.snapshot_control import Snapshots
from multiprocessing import Process
import time, cx_Oracle, csv, random
#
class Workload:
    #
    @staticmethod
    def execute_transaction(ev_loader, logger, transaction_path, query_stream, variant_path, outliers):
        """
        Wrapper method for method '__execute_and_forget'
        :param ev_loader: Environment context
        :param logger: Logger context
        :param transaction_path: Directory path to contained file
        :param transaction_name: File name containing TPC-DS transaction
        :param query_stream: List of queries ordered as indicated by stream_identification_number
        :return: Return slave process for barrier monitoring
        """
        slave_list = []
        for stream in query_stream:
            #logger.log('-----------------BUILDING_SLAVE-----------------')
            slave_list.append(Process(target=Workload.__execute_and_forget, args=(ev_loader, logger, transaction_path, stream, variant_path, outliers)))
        #
        for slave in slave_list:
            #logger.log('-----------------START_SLAVE-----------------')
            slave.start()
        #
        for slave in slave_list:
            #logger.log('-----------------WAITING FOR SLAVE TO JOIN-----------------')
            slave.join()
    #
    @staticmethod
    def execute_statistic_gatherer(ev_loader, logger, path_bank):
        """
        Wrapper method for '__statistic_gatherer'
        :param ev_loader: Environment context
        :param logger: Logger context
        :return:
        """
        p = Process(target=Workload.__statistic_gatherer, args=(ev_loader, logger, path_bank))
        p.start()
    #
    @staticmethod
    def __statistic_gatherer(ev_loader, logger, path_bank):
        """
        This method is tasked with polling the database instance and extract metric every N seconds. Extracted metrics
        are saved in .csv format on disk.

        The targetted tables for metric extraction are denoted below:
        * DBA_HIST_SQLSTAT
        * DBA_HIST_SYSMETRIC_SUMMARY
        * DBA_HIST_SYSSTAT
        * V$SQL_PLAN

        This method executes indefinitely, until the scheduler is terminated.
        :param ev_loader: Environment context
        :param logger: Logger context
        :param path_bank: List of csv files which will be written to during method invocation
        :return:
        """
        logger.log('Initiating statistic gatherer..')
        # kill_signal = 0
        query_sql_stat = "select dhsql.*, " \
                         "dhst.sql_text, " \
                         "dhst.command_type, " \
                         "dhsnap.startup_time, " \
                         "dhsnap.begin_interval_time, " \
                         "dhsnap.end_interval_time, " \
                         "dhsnap.flush_elapsed, " \
                         "dhsnap.snap_level, "\
                         "dhsnap.error_count, "\
                         "dhsnap.snap_flag, "\
                         "dhsnap.snap_timezone "\
                         "from dba_hist_sqlstat dhsql, "\
                         "dba_hist_snapshot dhsnap, "\
                         "dba_hist_sqltext dhst " \
                         "where dhsql.snap_id = dhsnap.snap_id "\
                         "and dhsql.dbid = dhsnap.dbid "\
                         "and dhsql.instance_number = dhsnap.instance_number "\
                         "and dhsql.dbid = dhst.dbid " \
                         "and dhsql.sql_id = dhst.sql_id " \
                         "and dhsnap.snap_id = :snap"
        # query_sql_plan = "select * " \
        #                  "from dba_hist_sql_plan vsp " \
        #                  "where vsp.sql_id in ( " \
        #                  "	select dhsql.sql_id " \
        #                  "	from dba_hist_sqlstat dhsql, " \
        #                  "	     dba_hist_snapshot dhsnap " \
        #                  "	where dhsql.snap_id = dhsnap.snap_id " \
        #                  "	and dhsql.dbid = dhsnap.dbid " \
        #                  "	and dhsql.instance_number = dhsnap.instance_number " \
        #                  "	and dhsnap.snap_id = :snap " \
        #                  ") " \
        #                  "and vsp.timestamp between ( " \
        #                  "  select max(BEGIN_INTERVAL_TIME) " \
        #                  "  from DBA_HIST_SNAPSHOT " \
        #                  "  where snap_id = :snap " \
        #                  ") and ( " \
        #                  "  select max(END_INTERVAL_TIME) " \
        #                  "  from DBA_HIST_SNAPSHOT " \
        #                  "  where snap_id = :snap " \
        #                  ") order by sql_id, id"
        query_sql_plan = "select dhs3.sql_text, " \
                            "       dhs4.sql_id, " \
                            "       dhs4.plan_hash_value, " \
                            "       dhs4.id, " \
                            "       dhs4.OPERATION, " \
                            "       dhs4.OPTIONS, " \
                            "       dhs4.OBJECT_NODE, " \
                            "       dhs4.OBJECT#, " \
                            "       dhs4.OBJECT_OWNER, " \
                            "       dhs4.OBJECT_NAME, " \
                            "       dhs4.OBJECT_ALIAS, " \
                            "       dhs4.OBJECT_TYPE, " \
                            "       dhs4.OPTIMIZER, " \
                            "       dhs4.PARENT_ID, " \
                            "       dhs4.depth, " \
                            "       dhs4.POSITION, " \
                            "       dhs4.SEARCH_COLUMNS, " \
                            "       dhs4.COST, " \
                            "       dhs4.CARDINALITY, " \
                            "       dhs4.BYTES, " \
                            "       dhs4.PARTITION_START, " \
                            "       dhs4.PARTITION_STOP, " \
                            "       dhs4.partition_id, " \
                            "       dhs4.DISTRIBUTION, " \
                            "       dhs4.CPU_COST, " \
                            "       dhs4.IO_COST, " \
                            "       dhs4.TEMP_SPACE, " \
                            "       dhs4.ACCESS_PREDICATES, " \
                            "       dhs4.FILTER_PREDICATES, " \
                            "       dhs4.projection, " \
                            "       dhs4.TIME, " \
                            "       dhs4.QBLOCK_NAME, " \
                            "       dhs4.TIMESTAMP " \
                            "from dba_hist_sqlstat dhs2, " \
                            "     dba_hist_sqltext dhs3, " \
                            "     dba_hist_sql_plan dhs4 " \
                            "where dhs2.sql_id = dhs3.sql_id " \
                            "and dhs2.dbid = dhs3.dbid " \
                            "and dhs2.snap_id = :snap " \
                            "and dhs2.DBID = (select max(dbid) from dba_hist_snapshot where snap_id = :snap ) " \
                            "and dhs2.parsing_schema_name = '" + ev_loader.var_get('user').upper() + "' " \
                            "and dhs3.sql_id = dhs4.sql_id " \
                            "and dhs3.dbid = dhs4.dbid " \
                            "and dhs3.CON_DBID = dhs4.CON_DBID " \
                            "order by dhs4.timestamp, " \
                            "         dhs4.plan_hash_value, " \
                            "         dhs4.id"
        query_hist_sysmetric_summary = "select dhss.*, " \
                                        "       dhsnap.startup_time, " \
                                        "       dhsnap.flush_elapsed, " \
                                        "       dhsnap.snap_level, " \
                                        "       dhsnap.error_count, " \
                                        "       dhsnap.snap_flag, " \
                                        "       dhsnap.snap_timezone " \
                                        "from DBA_HIST_SYSMETRIC_SUMMARY dhss, " \
                                        "     dba_hist_snapshot dhsnap " \
                                        "where dhss.snap_id = dhsnap.snap_id " \
                                        "and dhss.dbid = dhsnap.dbid " \
                                        "and dhss.instance_number = dhsnap.instance_number " \
                                        "and dhsnap.snap_id = :snap"
        query_hist_sysstat = "select dhsys.*, " \
                            "       dhs.startup_time, " \
                            "       dhs.begin_interval_time, " \
                            "       dhs.end_interval_time, " \
                            "       dhs.flush_elapsed, " \
                            "       dhs.snap_level, " \
                            "       dhs.error_count, " \
                            "       dhs.snap_flag, " \
                            "       dhs.snap_timezone " \
                            "from dba_hist_sysstat dhsys, " \
                            "     dba_hist_snapshot dhs " \
                            "where dhsys.snap_id = dhs.snap_id " \
                            "and dhsys.dbid = dhs.dbid " \
                            "and dhsys.instance_number = dhs.instance_number " \
                            "and dhs.snap_id = :snap "
        #
        try:
            #
            # Opens CSV file
            rep_hist_snapshot = open(path_bank[0], 'a')
            rep_sql_plan = open(path_bank[1], 'a')
            rep_hist_sysmetric_summary = open(path_bank[2], 'a')
            rep_hist_sysstat = open(path_bank[3],'a')
            #
            rep_hist_csv = csv.writer(rep_hist_snapshot, dialect='excel')
            rep_sql_csv = csv.writer(rep_sql_plan, dialect='excel')
            rep_hist_sysmetric_summary_csv = csv.writer(rep_hist_sysmetric_summary, dialect='excel')
            rep_hist_sysstat_csv = csv.writer(rep_hist_sysstat, dialect='excel')
            #
        except FileNotFoundError:
            logger.log('Statistic file was not found!')
            raise FileNotFoundError('Statistic file was not found!')
        except Exception as e:
            logger.log('An exception was raised during writing to file! [' + str(e) + ']')
        #
        # Creates database connection
        db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                                    user=ev_loader.var_get('user'),
                                    host=ev_loader.var_get('host'),
                                    service=ev_loader.var_get('service'),
                                    port=ev_loader.var_get('port'),
                                    password=ev_loader.var_get('password'),
                                    logger=logger)
        db_conn.connect()
        #
        # Create begin snapshot - This snap id is not saved to file since the time interval captured here is unknown.
        # Therefore this snap id establishes a base snapshot from which to base future snapshots.
        Snapshots.capture_snapshot(db_conn=db_conn, logger=logger)
        snap = Snapshots.get_max_snapid(db_conn=db_conn, logger=logger)
        #
        while True:
            #
            try:
                #
                # Wait N seconds
                time.sleep(ev_loader.var_get('statistic_intervals'))
                #
                # Create end snapshot
                Snapshots.capture_snapshot(db_conn=db_conn, logger=logger)
                snap = Snapshots.get_max_snapid(db_conn=db_conn, logger=logger)
                #
                logger.log('Polling metrics from dba_hist_sqlstat with snapid [' + str(snap) + '] ..')
                cur_hist_snapshot = db_conn.execute_query(query=query_sql_stat,
                                                          params={"snap":snap})
                logger.log('Polling metrics from v$sql_plan with snapid [' + str(snap) + '] ..')
                cur_sql_plan = db_conn.execute_query(query=query_sql_plan,
                                                     params={"snap":snap})
                logger.log('Polling metrics from dba_hist_sysmetric_summary with snapid [' + str(snap) + ']')
                cur_hist_sysmetric = db_conn.execute_query(query=query_hist_sysmetric_summary,
                                                           params={"snap":snap})
                logger.log('Polling metrics from dba_hist_sysstat with snapid [' + str(snap) + ']')
                cur_hist_sysstat = db_conn.execute_query(query=query_hist_sysstat,
                                                         params={"snap": snap})
                #
                # Write cursors to csv files
                [rep_hist_csv.writerow(row) for row in cur_hist_snapshot]
                [rep_sql_csv.writerow(row) for row in cur_sql_plan]
                [rep_hist_sysmetric_summary_csv.writerow(row) for row in cur_hist_sysmetric]
                [rep_hist_sysstat_csv.writerow(row) for row in cur_hist_sysstat]
                logger.log('Metrics successfully written to file..')
                #
            except Exception as e:
                logger.log('An exception was caught in method ''__statistic_gatherer'' [' + str(e) + ']')
                db_conn.close()
                break
        #         kill_signal = 1
        #     #
        #     if kill_signal > 0:
        #         break
        # #
        # # Closes csv file/s
        # rep_hist_snapshot.close()
        # rep_sql_plan.close()
        # rep_hist_sysmetric_summary.close()
        # rep_hist_sysstat.close()
        # #
        # # Closes database connection
        # db_conn.close() # This line most will most likely not be needed given that the database would have just restarted, and bounded all connections
        # logger.log('Killed statistic gatherer..')
    #
    @staticmethod
    def __execute_and_forget(ev_loader, logger, transaction_path, query_stream, variant_path, outliers):
        """
        This method executes a TPC-DS transaction (query/dml), and left to finish.

        This method is designed to be executed and forgotten. Once executed, this child will no longer be controlled by the
        driver.
        :param ev_loader: Environment context
        :param logger: Logger context
        :param transaction_path: Directory path to contained file
        :param transaction_name: File name containing TPC-DS transaction
        :param query_stream: List of queries ordered as indicated by stream_identification_number
        :param variant_path: File path corresponding to variant TPC-DS Queries
        :param outliers: Outlier threshold (Percentage)
        :return:
        """
        for query_id in query_stream:
            #
            path = transaction_path + 'query_' + str(query_id) + '.sql'
            #
            if int(query_id) in outliers and random.random() > ev_loader.var_get('outlier_threshold'):
                path = variant_path + 'query_' + str(query_id) + '.sql'
            #
            DatabaseInterface.execute_script(user=ev_loader.var_get('user'),
                                             password=ev_loader.var_get('password'),
                                             instance_name=ev_loader.var_get('instance_name'),
                                             filename=path,
                                             params=None,
                                             logger=logger,
                                             redirect_path=ev_loader.var_get('project_dir') + "/log/sqlplusoutput.txt")
    #
    @staticmethod
    def get_script_headers(report_type=None, ev_loader=None, logger=None):
        """
        Retrieves column headers to populate csv reports (containing the extracted metrics)
        :param report_type: Currently supports two report types: 'rep_hist_snapshot','rep_vsql_plan'
        :param ev_loader: Environment context
        :param logger: Logger context
        :return:
        """
        if report_type is None:
            raise ValueError('Report Type was not specified!')
        #
        if report_type == 'rep_hist_snapshot':
            query = "select column_name " \
                    "from ( " \
                    "select table_name, column_name, column_id, 1 as table_order " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SQLSTAT' " \
                    "union all " \
                    "select table_name, column_name, column_id, 2 as table_order " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SQLTEXT' " \
                    "and column_name in ('SQL_TEXT', " \
                    "                    'COMMAND_TYPE') " \
                    "union all " \
                    "select table_name, column_name, column_id, 3 as table_order " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SNAPSHOT' " \
                    "and column_name in ('STARTUP_TIME', " \
                    "					'BEGIN_INTERVAL_TIME', " \
                    "					'END_INTERVAL_TIME', " \
                    "					'FLUSH_ELAPSED', " \
                    "					'SNAP_LEVEL', " \
                    "					'ERROR_COUNT', " \
                    "					'SNAP_FLAG', " \
                    "					'SNAP_TIMEZONE') " \
                    ") order by table_order asc, " \
                    "		   column_id asc "
        elif report_type == 'rep_vsql_plan':
            # query = "select column_name " \
            #         "from all_tab_columns " \
            #         "where table_name = 'DBA_HIST_SQL_PLAN' " \
            #         "order by column_id"
            query = 'select "sql_text", ' \
                    '       "sql_id", ' \
                    '       "plan_hash_value", ' \
                    '       "id", ' \
                    '       "operation", ' \
                    '       "options", ' \
                    '       "object_node", ' \
                    '       "object#", ' \
                    '       "object_owner", ' \
                    '       "object_name", ' \
                    '       "object_alias", ' \
                    '       "object_type", ' \
                    '       "optimizer", ' \
                    '       "parent_id", ' \
                    '       "depth", ' \
                    '       "position", ' \
                    '       "search_columns", ' \
                    '       "cost", ' \
                    '       "cardinality", ' \
                    '       "bytes", ' \
                    '       "partition_start", ' \
                    '       "partition_stop", ' \
                    '       "partition_id", ' \
                    '       "distribution", ' \
                    '       "cpu_cost", ' \
                    '       "io_cost", ' \
                    '       "temp_space", ' \
                    '       "access_predicates", ' \
                    '       "filter_predicates", ' \
                    '       "projection", ' \
                    '       "time", ' \
                    '       "qblock_name", ' \
                    '       "time_stamp" from dual'
            logger.log(query)
        elif report_type == 'rep_hist_sysmetric_summary':
            query = "select column_name " \
                    "from ( " \
                    "select table_name, column_name, column_id " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SYSMETRIC_SUMMARY' " \
                    "union all " \
                    "select table_name, column_name, column_id " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SNAPSHOT' " \
                    "and column_name in ('STARTUP_TIME', " \
                    "					'BEGIN_INTERVAL_TIME', " \
                    "					'END_INTERVAL_TIME', " \
                    "					'FLUSH_ELAPSED', " \
                    "					'SNAP_LEVEL', " \
                    "					'ERROR_COUNT', " \
                    "					'SNAP_FLAG', " \
                    "					'SNAP_TIMEZONE') " \
                    ") order by table_name desc, " \
                    "		   column_id asc"
        elif report_type == 'rep_hist_sysstat':
            query = "select column_name " \
                    "from ( " \
                    "select table_name, column_name, column_id " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SYSSTAT' " \
                    "union all " \
                    "select table_name, column_name, column_id " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SNAPSHOT' " \
                    "and column_name in ('STARTUP_TIME', " \
                    "					'BEGIN_INTERVAL_TIME', " \
                    "					'END_INTERVAL_TIME', " \
                    "					'FLUSH_ELAPSED', " \
                    "					'SNAP_LEVEL', " \
                    "					'ERROR_COUNT', " \
                    "					'SNAP_FLAG', " \
                    "					'SNAP_TIMEZONE') " \
                    ") order by table_name desc, " \
                    "		   column_id asc"
        elif report_type == 'rep_execution_plans':
            query = "select * " \
                    "from ( " \
                    "select column_name " \
                    "from dba_tab_columns " \
                    "where table_name = 'V_$SQL' " \
                    "order by column_id " \
                    ") t " \
                    "union all " \
                    "select 'TPC_TRANSACTION_NAME' " \
                    "from dual " \
                    "union all " \
                    "select 'STATEMENT_HASH_SUM' " \
                    "from dual " \
                    "union all " \
                    "select 'BENCHMARK_ITERATION' " \
                    "from dual " \
                    "union all " \
                    "select 'GATHERED_STATS' " \
                    "from dual"
        elif report_type == 'rep_explain_plans':
            query = "select * " \
                    "from ( " \
                    "select column_name " \
                    "from dba_tab_columns " \
                    "where table_name = 'PLAN_TABLE$' " \
                    "and column_name in ('STATEMENT_ID', " \
                    "					'PLAN_ID', " \
                    "					'TIMESTAMP', " \
                    "					'REMARKS', " \
                    "					'OPERATION', " \
                    "					'OPTIONS', " \
                    "					'OBJECT_NODE', " \
                    "					'OBJECT_OWNER', " \
                    "					'OBJECT_NAME', " \
                    "					'OBJECT_INSTANCE', " \
                    "					'OBJECT_TYPE', " \
                    "					'OPTIMIZER', " \
                    "					'SEARCH_COLUMNS', " \
                    "					'ID', " \
                    "					'PARENT_ID', " \
                    "					'DEPTH', " \
                    "					'POSITION', " \
                    "					'COST', " \
                    "					'CARDINALITY', " \
                    "					'BYTES', " \
                    "					'OTHER_TAG', " \
                    "					'PARTITION_START', " \
                    "					'PARTITION_STOP', " \
                    "					'PARTITION_ID', " \
                    "					'DISTRIBUTION', " \
                    "					'CPU_COST', " \
                    "					'IO_COST', " \
                    "					'TEMP_SPACE', " \
                    "					'ACCESS_PREDICATES', " \
                    "					'FILTER_PREDICATES', " \
                    "					'TIME') " \
                    "order by column_id " \
                    ") t " \
                    "union all " \
                    "select 'TPC_TRANSACTION_NAME' " \
                    "from dual " \
                    "union all " \
                    "select 'STATEMENT_HASH_SUM' " \
                    "from dual " \
                    "union all " \
                    "select 'BENCHMARK_ITERATION' " \
                    "from dual " \
                    "union all " \
                    "select 'GATHERED_STATS' " \
                    "from dual"
        else:
            raise ValueError('Unsupported type!')
        #
        db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                                    user=ev_loader.var_get('user'),
                                    host=ev_loader.var_get('host'),
                                    service=ev_loader.var_get('service'),
                                    port=ev_loader.var_get('port'),
                                    password=ev_loader.var_get('password'),
                                    logger=logger)
        db_conn.connect()
        res = db_conn.execute_query(query=query,params=None)
        db_conn.close()
        column_list = []
        for row in res:
            column_list.append(row)
        return column_list