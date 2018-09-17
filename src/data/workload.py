#
# Module Imports
from src.framework.db_interface import DatabaseInterface
from src.utils.snapshot_control import Snapshots
from multiprocessing import Process
import time, cx_Oracle, csv
#
class Workload:
    #
    __active_thread_count = 0 # Denotes how many active threads are running, to avoid resource starvation
    #
    @staticmethod
    def execute_transaction(ev_loader, logger, transaction_path, transaction_name):
        """
        Wrapper method for method '__execute_and_forget'
        :param ev_loader:
        :param logger:
        :param transaction_path:
        :param transaction_name:
        :return:
        """
        p = Process(target=Workload.__execute_and_forget, args=(ev_loader, logger, transaction_path, transaction_name))
        p.start()
    #
    @staticmethod
    def execute_statistic_gatherer(ev_loader, logger, path_bank):
        """
        Wrapper method for '__statistic_gatherer'
        :param ev_loader:
        :param logger:
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

        This method executes indefinitely, until the scheduler is terminated.
        :param ev_loader:
        :param logger:
        :param path_bank
        :return:
        """
        logger.log('Initiating statistic gatherer..')
        Workload.__active_thread_count += 1
        kill_signal = 0
        query_sql_stat = "select dhsql.*, " \
                         "dhsnap.startup_time, " \
                         "dhsnap.begin_interval_time, " \
                         "dhsnap.end_interval_time, " \
                         "dhsnap.flush_elapsed, " \
                         "dhsnap.snap_level, "\
                         "dhsnap.error_count, "\
                         "dhsnap.snap_flag, "\
                         "dhsnap.snap_timezone "\
                         "from dba_hist_sqlstat dhsql, "\
                         "dba_hist_snapshot dhsnap "\
                         "where dhsql.snap_id = dhsnap.snap_id "\
                         "and dhsql.dbid = dhsnap.dbid "\
                         "and dhsql.instance_number = dhsnap.instance_number "\
                         "and dhsnap.snap_id between :snap_begin and :snap_end"
        query_sql_plan = "select * " \
                         "from v$sql_plan vsp " \
                         "where vsp.sql_id in ( " \
                         "	select dhsql.sql_id " \
                         "	from dba_hist_sqlstat dhsql, " \
                         "	     dba_hist_snapshot dhsnap " \
                         "	where dhsql.snap_id = dhsnap.snap_id " \
                         "	and dhsql.dbid = dhsnap.dbid " \
                         "	and dhsql.instance_number = dhsnap.instance_number " \
                         "	and dhsnap.snap_id between :snap_begin and :snap_end " \
                         ") " \
                         "and vsp.timestamp = ( " \
                         "  select max(timestamp) " \
                         "  from v$sql_plan " \
                         "  where sql_id = vsp.sql_id " \
                         ") " \
                         "order by sql_id, id"
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
                                        "and dhsnap.snap_id between :snap_begin and :snap_end"
        #
        # Opens CSV file
        try:
            rep_hist_snapshot = open(path_bank[0], 'a')
            rep_sql_plan = open(path_bank[1], 'a')
            rep_hist_sysmetric_summary = open(path_bank[2], 'a')
            rep_hist_csv = csv.writer(rep_hist_snapshot, dialect='excel')
            rep_sql_csv = csv.writer(rep_sql_plan, dialect='excel')
            rep_hist_sysmetric_summary_csv = csv.writer(rep_hist_sysmetric_summary, dialect='excel')
        except FileNotFoundError:
            logger.log('REP_HIST_SNAPSHOT.csv was not found!')
            raise FileNotFoundError('REP_HIST_SNAPSHOT.csv was not found!')
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
        while True:
            #
            try:
                #
                # Create begin snapshot
                Snapshots.capture_snapshot(db_conn=db_conn, logger=logger)
                snap_begin = Snapshots.get_max_snapid(db_conn=db_conn, logger=logger)
                #
                # Wait N seconds
                time.sleep(ev_loader.var_get('statistic_intervals'))
                #
                # Create end snapshot
                Snapshots.capture_snapshot(db_conn=db_conn, logger=logger)
                snap_end = Snapshots.get_max_snapid(db_conn=db_conn, logger=logger)
                #
                logger.log('Polling metrics from dba_hist_sqlstat between SNAPIDs [' + str(snap_begin) + ',' + str(snap_end) + '] ..')
                cur_hist_snapshot = db_conn.execute_query(query=query_sql_stat,
                                                          params={"snap_begin":snap_begin,"snap_end":snap_end})
                logger.log('Polling metrics from v$sql_plan between SNAPIDs [' + str(snap_begin) + ',' + str(snap_end) + '] ..')
                cur_sql_plan = db_conn.execute_query(query=query_sql_plan,
                                                     params={"snap_begin":snap_begin,"snap_end":snap_end})
                logger.log('Polling metrics from dba_hist_sysmetric_summary')
                cur_hist_sysmetric = db_conn.execute_query(query=query_hist_sysmetric_summary,
                                                           params={"snap_begin":snap_begin,"snap_end":snap_end})
                #
                # Write cursors to csv files
                [rep_hist_csv.writerow(row) for row in cur_hist_snapshot]
                [rep_sql_csv.writerow(row) for row in cur_sql_plan]
                [rep_hist_sysmetric_summary_csv.writerow(row) for row in cur_hist_sysmetric]
                logger.log('Metrics successfully written to file..')
                #
            except Exception as e:
                logger.log('An exception was caught in method ''__statistic_gatherer'' [' + str(e) + ']')
                kill_signal = 1
            #
            if kill_signal > 0:
                break
        #
        # Closes csv file/s
        rep_hist_csv.close()
        rep_sql_csv.close()
        rep_hist_sysmetric_summary_csv.close()
        #
        # Closes database connection
        db_conn.close() # This line most will most likely not be needed given that the database would have just restarted, and bounded all connections
        Workload.__active_thread_count -= 1
        logger.log('Killed statistic gatherer..')
    #
    @staticmethod
    def __execute_and_forget(ev_loader, logger, transaction_path, transaction_name):
        """
        This method executes a TPC-DS transaction (query/dml), and left to finish.

        This method is designed to be executed and forgotten. Once executed, this child will no longer be controlled by the
        driver.
        :param ev_loader:
        :param logger:
        :param transaction_path:
        :param transaction_name:
        :return:
        """
        Workload.__active_thread_count += 1
        #
        start_time = time.clock()
        DatabaseInterface.execute_script(user=ev_loader.var_get('user'),
                                         password=ev_loader.var_get('password'),
                                         instance_name=ev_loader.var_get('instance_name'),
                                         filename=transaction_path + transaction_name,
                                         params=None,
                                         logger=logger,
                                         redirect_path=ev_loader.var_get('project_dir') + "/log/sqlplusoutput.txt")
        end_time = time.clock() - start_time
        logger.log('Successfully executed ' + transaction_name + " under " + str(end_time) + " seconds.")
        #
        Workload.__active_thread_count -= 1

    #
    @staticmethod
    def barrier(ev_loader):
        """
        Halts driver from proceeding any further if active thread count is greater than 'parallel_cap'
        :param ev_loader:
        :return:
        """
        while True:
            if Workload.__active_thread_count < ev_loader.var_get('parallel_cap'):
                break
            else:
                time.sleep(4)
    #
    @staticmethod
    def get_script_headers(report_type=None, ev_loader=None, logger=None):
        """
        Retrieves column headers to populate csv reports (containing the extracted metrics)
        :param report_type: Currently supports two report types: 'rep_hist_snapshot','rep_vsql_plan'
        :param ev_loader:
        :param logger:
        :return:
        """
        if report_type is None:
            raise ValueError('Report Type was not specified!')
        #
        if report_type == 'rep_hist_snapshot':
            query = "select column_name " \
                    "from ( " \
                    "select table_name, column_name, column_id " \
                    "from dba_tab_columns " \
                    "where table_name = 'DBA_HIST_SQLSTAT' " \
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
        elif report_type == 'rep_vsql_plan':
            query = "select column_name " \
                    "from all_tab_columns " \
                    "where table_name = 'V_$SQL_PLAN' " \
                    "order by column_id"
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