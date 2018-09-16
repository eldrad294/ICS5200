#
class Snapshots:
    #
    @staticmethod
    def capture_snapshot(db_conn, logger):
        Snapshots.__validate(db_conn=db_conn,
                             logger=logger)
        #
        db_conn.execute_proc(name='DBMS_WORKLOAD_REPOSITORY.CREATE_SNAPSHOT',
                             parameters={})
    #
    @staticmethod
    def get_max_snapid(db_conn, logger):
        Snapshots.__validate(db_conn=db_conn,
                             logger=logger)
        #
        max_snapshot_query = "select max(snap_id) " \
                             "from dba_hist_snapshot"
        snap_id = db_conn.execute_query(query=max_snapshot_query,
                                        fetch_single=True)
        return snap_id[0]
    #
    @staticmethod
    def __validate(db_conn, logger):
        if db_conn is None:
            raise Exception('Database connection was not established!')
        if logger is None:
            raise Exception('Logger object was not established!')