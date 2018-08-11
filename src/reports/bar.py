from plotly.offline import plot
import plotly.graph_objs as go
from plotly.graph_objs import *
import src.reports.report_extraction_sql as res
#
class BarCharts:
    #
    def __init__(self, db_conn, logger):
        self.__db_conn = db_conn
        self.__logger = logger
    #
    def generate_REP_TPC_DESCRIBE(self):
        """
        Generates the REP_TPC_DESCRIBE.sql reprot
        :return:
        """
        self.__logger.log('Starting generation of report..')
        #
        cur, schema = self.__db_conn.execute_query(res.REP_TPC_DESCRIBE, describe=True)
        #
        print(schema)
        print(cur)
        self.__logger.log('Report generation complete')
