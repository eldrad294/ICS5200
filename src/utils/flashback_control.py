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
        return str(FlashbackControl.__getTimeStamp())
    #
    @staticmethod
    def flashback_tables(db_conn, logger, timestamp, ev_loader):
        """
        Flashbacks all tables located within user_tables to ensure that changes carried out by the flashback are
        restored.
        :param db_conn: Database Connection
        :param logger: Logger Instance
        :param timestamp: Timstamp of flashback point
        :param ev_loader: environment data instance
        :return:
        """
        logger.log('Enabling Table Flashback..')
        sql = "select table_name from user_tables where tablespace_name in ('" + str(ev_loader.var_get('user')).upper() \
              + "')"
        res = db_conn.execute_query(query=sql, describe=False)
        for table in res:
            table = table[0]
            sql = "flashback table " + table + " to timestamp to_date('" + timestamp + "','DD-MM-YYYY HH24:MI:SS')"
            logger.log('Flashing table [' + table + '] to established timestamp [' + timestamp + ']')
            db_conn.execute_dml(dml=sql)
    #
    @staticmethod
    def __getTimeStamp():
        """
        :return: Returns system timestamp
        """
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y %H:%M:%S')