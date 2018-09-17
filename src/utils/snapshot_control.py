#
class Snapshots:
    """
    Control class / wrapper around database snapshot invocation
    """
    #
    @staticmethod
    def capture_snapshot(db_conn, logger):
        """
        Captures snapshot at instance level - this will hold a number of statistical information at the database point
        in time
        :param db_conn: Connection context
        :param logger: Logger context
        :return:
        """
        Snapshots.__validate(db_conn=db_conn,
                             logger=logger)
        #
        db_conn.execute_proc(name='DBMS_WORKLOAD_REPOSITORY.CREATE_SNAPSHOT',
                             parameters={})
    #
    @staticmethod
    def get_max_snapid(db_conn, logger):
        """
        Returns largest snap id from DBA_HIST_SNAPSHOT
        :param db_conn: Connection context
        :param logger: Logger context
        :return:
        """
        Snapshots.__validate(db_conn=db_conn,
                             logger=logger)
        #
        max_snapshot_query = "select snap_id " \
                             "from dba_hist_snapshot " \
                             "where end_interval_time = select max(end_interval_time) from dba_hist_snapshot"
        snap_id = db_conn.execute_query(query=max_snapshot_query,
                                        fetch_single=True)
        return snap_id[0]
    #
    @staticmethod
    def __validate(db_conn, logger):
        """
        Validation method
        :param db_conn:
        :param logger:
        :return:
        """
        if db_conn is None:
            raise Exception('Database connection was not established!')
        if logger is None:
            raise Exception('Logger object was not established!')