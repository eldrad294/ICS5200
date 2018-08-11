from plotly.offline import plot
import plotly.graph_objs as go
from plotly.graph_objs import *
import src.reports.report_extraction_sql as res
#
class BarCharts:
    #
    def __init__(self, db_conn):
        self.__db_conn = db_conn
    #
    def generate_REP_TPC_DESCRIBE(self):
        """
        Generates the REP_TPC_DESCRIBE.sql reprot
        :return:
        """
        logger.log('Starting generation of report..')
        #
        cur = self.__db_conn.execute_query(res.REP_TPC_DESCRIBE, describe=True)
        #
        print(cur)