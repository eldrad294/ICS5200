from plotly.offline import plot
from plotly.graph_objs import *
import plotly.graph_objs as go
import os
#
class BarCharts:
    #
    def __init__(self, db_conn, logger, save_path):
        """
        This constructor should eventually move to a parent class, to allow report generation classes to inherit config
        from a single class
        :param db_conn:
        :param logger:
        :param save_path:
        """
        self.__db_conn = db_conn
        self.__logger = logger
        self.__save_path = save_path
        #
        # Checks if report generation directory exists, otherwise creates it
        if os.path.isdir(self.__save_path) is False:
            os.mkdir(self.__save_path)
    #
    def generate_REP_TPC_DESCRIBE(self, tpc_type='tpcds1'):
        """
        Generates the REP_TPC_DESCRIBE.sql reprot
        :return:
        """
        self.__logger.log('Starting generation of report..')
        #
        cur, schema = self.__db_conn.execute_query(query="select * from REP_TPC_DESCRIBE where tpctype='" + tpc_type.upper() + "'",
                                                   describe=True)
        #
        # print(schema)
        # print(cur)
        table_name, row_count, index_count = [], [], []
        for row in cur:
            table_name.append(row[1])
            row_count.append(row[2])
            index_count.append(row[3])
        #
        data = Data([
            Bar(
                x=table_name,
                y=row_count,
                name='Row Count'
            ),
            Bar(x=table_name,
                y=index_count,
                name='Index Count')
        ])
        layout = go.Layout(
            barmode='group',
            title=tpc_type.upper() + " Description"
        )
        config = None
        fig = go.Figure(data=data, layout=layout)
        plot(fig, config=config, filename=self.__save_path + "/REP_TPC_DESCRIBE_" + str(tpc_type) + ".html", auto_open=False)
        #
        self.__logger.log('Report generation complete')
