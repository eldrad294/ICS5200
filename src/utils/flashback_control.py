#
# Import Modules
import datetime, time
#
class FlashbackControl:
    #
    @staticmethod
    def captureTimeStamp():
        """
        :return: Retrieve point in time
        """
        return FlashbackControl.__getTimeStamp()
    #
    @staticmethod
    def flashback_tables(db_conn, logger, timestamp, ev_loader):
        """
        Flashbacks all tables located within user_tables to ensure that changes carried out by the flashback are
        restored.
        :param db_conn:
        :param logger:
        :param timestamp:
        :param ev_loader:
        :return:
        """
        logger.log('Enabling Table Flashback..')
        sql = "select table_name from user_tables where tablespace_name in ('" + str(ev_loader.var_get('user')).upper() \
              + "','TPCDS_BENCHMARK')"
        res = db_conn.execute_query(query=sql, describe=False)
        for table in res:
            sql = "flashback table " + table + " to timestamp to_date('" + timestamp + "','DD-MM-YYYY HH24:MI:SS')"
            logger.log('Flashing table [' + table + '] to established timestamp [' + timestamp + ']')
            db_conn.execute_dml(dml=sql)
    #
    @staticmethod
    def __getTimeStamp(self):
        """
        :return: Returns system timestamp
        """
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y %H:%M:%S')