#
class OptimizerStatistics:
    #
    @staticmethod
    def generate_optimizer_statistics(db_conn, logger, tpctype=None):
        #
        OptimizerStatistics.__validate(db_conn=db_conn,
                                       logger=logger,
                                       tpc_type=tpctype)
        #
        params = {"statown": tpctype.upper(),
                  "estimate_percent":"DBMS_STATS.AUTO_SAMPLE_SIZE",
                  "degree":60,
                  "granularity":'ALL',
                  "cascade":'TRUE',
                  "method_opt":'FOR ALL COLUMNS',
                  "options":'GATHER',
                  "gather_sys":'TRUE',
                  "no_invalidate":'DBMS_STATS.AUTO_INVALIDATE'}
        db_conn.execute_proc(name='dbms_stats.gather_database_stats',
                             parameters=params)
    #
    @staticmethod
    def remove_optimizer_statistics(db_conn, logger, tpctype=None):
        OptimizerStatistics.__validate(db_conn=db_conn,
                                       logger=logger,
                                       tpc_type=tpctype)
        #
        params = {"statown":tpctype.upper()}
        db_conn.execute_proc(name='dbms_stats.delete_database_stats',
                             parameters=params)
    #
    @staticmethod
    def __validate(db_conn, logger, tpc_type):
        if db_conn is None:
            raise Exception('Database connection was not established!')
        if logger is None:
            raise Exception('Logger object was not established!')
        if tpc_type is None:
            raise ValueError('TPC Type was not declared!')
