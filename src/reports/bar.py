from plotly.offline import plot
from plotly.graph_objs import *
from src.utils.plan_control import XPlan
import plotly.graph_objs as go
import os, numpy as np, pandas as pd
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
        Generates the REP_TPC_DESCRIBE.sql report
        :param tpc_type:
        :return:
        """
        self.__logger.log('Starting generation of report..')
        #
        cur, schema = self.__db_conn.execute_query(query="select * from REP_TPC_DESCRIBE",
                                                   describe=True)
        #
        # print(schema)
        # print(cur)
        table_name, row_count, index_count = [], [], []
        for row in cur:
            table_name.append(row[0])
            row_count.append(row[1])
            index_count.append(row[2])
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
    #
    def generate_REP_EXECUTION_PLANS(self, ev_loader=None, gathered_stats=False, iterations=5, columns=None, from_table=False):
        """
        Generates the REP_EXECUTION_PLANS.sql report
        :param tpc_type: TPC-DSX
        :param gathered_stats: True or False, depending on the type of benchmark
        :param iterations: Number of expected iterations
        :param columns: User specification as to which columns to plot. Graph will always be plot against
                        tpc_transaction_name on X axis.
        :param from_table: If 'True', query instance and retrieve data. If 'False', retrieve data from csv under /src/sql/Runtime/TPC-DS/tpcds1/Benchmark/*
        :return:
        """
        self.__logger.log('Starting generation of report..')
        #
        if columns is None or len(columns) < 1:
            raise ValueError('No columns were assigned to be plotted!')
        columns = [column.upper() for column in columns]
        #
        if from_table:
            query = " select tpc_transaction_name as tpc_transaction_name, " \
                    " count(tpc_transaction_name) as number_of_executions, " \
                    " sum(SHARABLE_MEM) / " + str(iterations) + " as SHARABLE_MEM, " \
                    " sum(PERSISTENT_MEM) / " + str(iterations) + " as PERSISTENT_MEM, " \
                    " sum(RUNTIME_MEM) / " + str(iterations) + " as RUNTIME_MEM, " \
                    " sum(SORTS) / " + str(iterations) + " as SORTS, " \
                    " sum(LOADED_VERSIONS) / " + str(iterations) + " as LOADED_VERSIONS, " \
                    " sum(OPEN_VERSIONS) / " + str(iterations) + " as OPEN_VERSIONS, " \
                    " sum(USERS_OPENING) / " + str(iterations) + " as USERS_OPENING, " \
                    " sum(FETCHES) / " + str(iterations) + " as FETCHES, " \
                    " sum(EXECUTIONS) / " + str(iterations) + " as EXECUTIONS, " \
                    " sum(PX_SERVERS_EXECUTIONS) / " + str(iterations) + " as PX_SERVERS_EXECUTIONS, " \
                    " sum(END_OF_FETCH_COUNT) / " + str(iterations) + " as END_OF_FETCH_COUNT, " \
                    " sum(USERS_EXECUTING) / " + str(iterations) + " as USERS_EXECUTING, " \
                    " sum(LOADS) / " + str(iterations) + " as LOADS, " \
                    " min(FIRST_LOAD_TIME) as FIRST_LOAD_TIME, " \
                    " sum(INVALIDATIONS) / " + str(iterations) + " as INVALIDATIONS, " \
                    " sum(PARSE_CALLS) / " + str(iterations) + " as PARSE_CALLS, " \
                    " sum(DISK_READS) / " + str(iterations) + " as DISK_READS, " \
                    " sum(DIRECT_WRITES) / " + str(iterations) + " as DIRECT_WRITES, " \
                    " sum(BUFFER_GETS) / " + str(iterations) + " as BUFFER_GETS, " \
                    " sum(APPLICATION_WAIT_TIME) / " + str(iterations) + " as APPLICATION_WAIT_TIME, " \
                    " sum(CONCURRENCY_WAIT_TIME) / " + str(iterations) + " as CONCURRENCY_WAIT_TIME, " \
                    " sum(CLUSTER_WAIT_TIME) / " + str(iterations) + " as CLUSTER_WAIT_TIME, " \
                    " sum(USER_IO_WAIT_TIME) / " + str(iterations) + " as USER_IO_WAIT_TIME, " \
                    " round((sum(PLSQL_EXEC_TIME) / " + str(iterations) + ") / (1000*60*60)) as PLSQL_EXEC_TIME_MINS, " \
                    " sum(JAVA_EXEC_TIME) / " + str(iterations) + " as JAVA_EXEC_TIME, " \
                    " sum(OPTIMIZER_COST) / " + str(iterations) + " as OPTIMIZER_COST, " \
                    " sum(CHILD_NUMBER) / " + str(iterations) + " as CHILD_NUMBER, " \
                    " sum(SERIALIZABLE_ABORTS) / " + str(iterations) + " as SERIALIZABLE_ABORTS, " \
                    " sum(OUTLINE_CATEGORY) / " + str(iterations) + " as OUTLINE_CATEGORY, " \
                    " round((sum(CPU_TIME) / " + str(iterations) + ") / (1000*60*60)) as CPU_TIME_MINS, " \
                    " round((sum(ELAPSED_TIME) / " + str(iterations) + ") / (1000*60*60)) as ELAPSED_TIME_MINS, " \
                    " sum(OUTLINE_SID) / " + str(iterations) + " as OUTLINE_SID, " \
                    " sum(SQLTYPE) / " + str(iterations) + " as SQLTYPE, " \
                    " min(LAST_LOAD_TIME) as LAST_LOAD_TIME, " \
                    " sum(CHILD_LATCH) / " + str(iterations) + " as CHILD_LATCH, " \
                    " min(LAST_ACTIVE_TIME) as LAST_ACTIVE_TIME, " \
                    " sum(TYPECHECK_MEM) / " + str(iterations) + " as TYPECHECK_MEM, " \
                    " sum(IO_CELL_OFFLOAD_ELIGIBLE_BYTES) / " + str(iterations) + " as IO_CELL_OFFLOAD_ELIGIBLE_BYTES, " \
                    " sum(IO_INTERCONNECT_BYTES) / " + str(iterations) + " as IO_INTERCONNECT_BYTES, " \
                    " sum(PHYSICAL_READ_REQUESTS) / " + str(iterations) + " as PHYSICAL_READ_REQUESTS, " \
                    " sum(PHYSICAL_READ_BYTES) / " + str(iterations) + " as PHYSICAL_READ_BYTES, " \
                    " sum(PHYSICAL_WRITE_REQUESTS) / " + str(iterations) + " as PHYSICAL_WRITE_REQUESTS, " \
                    " sum(PHYSICAL_WRITE_BYTES) / " + str(iterations) + " as PHYSICAL_WRITE_BYTES, " \
                    " sum(OPTIMIZED_PHY_READ_REQUESTS) / " + str(iterations) + " as OPTIMIZED_PHY_READ_REQUESTS, " \
                    " sum(LOCKED_TOTAL) / " + str(iterations) + " as LOCKED_TOTAL, " \
                    " sum(PINNED_TOTAL) / " + str(iterations) + " as PINNED_TOTAL, " \
                    " sum(IO_CELL_UNCOMPRESSED_BYTES) / " + str(iterations) + " as IO_CELL_UNCOMPRESSED_BYTES, " \
                    " sum(IO_CELL_OFFLOAD_RETURNED_BYTES) / " + str(iterations) + " as IO_CELL_OFFLOAD_RETURNED_BYTES, " \
                    " sum(IM_SCANS) / " + str(iterations) + " as IM_SCANS, " \
                    " sum(IM_SCAN_BYTES_UNCOMPRESSED) / " + str(iterations) + " as IM_SCAN_BYTES_UNCOMPRESSED, " \
                    " sum(IM_SCAN_BYTES_INMEMORY) / " + str(iterations) + " as IM_SCAN_BYTES_INMEMORY, " \
                    " count(STATEMENT_HASH_SUM) as STATEMENT_HASH_SUM, " \
                    " count(BENCHMARK_ITERATION) as BENCHMARK_ITERATIONS " \
                    " from REP_EXECUTION_PLANS " \
                    " where GATHERED_STATS = '" + str(gathered_stats).title() + "' " \
                    " group by tpc_transaction_name " \
                    " order by first_load_time, " \
                    " tpc_transaction_name"
            #print(query)
            cur, schema = self.__db_conn.execute_query(query=query,
                                                       describe=True)
            transaction_bank = []
            for row in cur:
                transaction_bank.append(row[0])
            #
            # print(schema)
            # print(cur)
            # print(np.array(cur)[:,0])
            for col in columns:
                for i in range(len(schema)):
                    if col == schema[i]:
                        data = Data([
                            Bar(
                                x=transaction_bank,
                                y=np.array(cur)[:,i],
                                name=col # Bar Title
                            )
                        ])
                        layout = go.Layout(
                            barmode='group',
                            title=ev_loader.var_get('user').upper() + " Benchmark " + str(col)
                        )
                        config = None
                        fig = go.Figure(data=data, layout=layout)
                        save_path = "/REP_EXECUTION_PLANS_" + str(ev_loader.var_get('user')) + '_' + str(gathered_stats) + '_' + col + '.html'
                        plot(fig, config=config, filename=self.__save_path + save_path, auto_open=False)
                self.__logger.log(str(col) + ' graph generated successfully..')
        else:
            csv_rep_execution_plans = "/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_execution_plans.csv"
            df = pd.read_csv(csv_rep_execution_plans)
            #
            # Check whether schema needs creating - executed only if relevant tables are not found
            sql_statement = "select count(*) from user_tables where table_name = 'REP_EXECUTION_PLANS'"
            result = int(self.__db_conn.execute_query(sql_statement, fetch_single=True)[0])
            if result == 0:
                #
                # Create metric table
                xp = XPlan(logger=self.__logger,
                           ev_loader=ev_loader)
                xp.create_REP_EXECUTION_PLANS(db_conn=self.__db_conn)
            #
            cur, schema = self.__db_conn.execute_query(query='select * from rep_execution_plans', describe=True)
            transaction_bank = []
            [transaction_bank.append(row[0]) for row in cur]
            #
            for col in columns:
                data = Data([
                    Bar(
                        x=transaction_bank,
                        y=df[col].loc[df[col] == str(gathered_stats)].values.tolist(),
                        name=col  # Bar Title
                    )
                ])
                layout = go.Layout(
                    barmode='group',
                    title=ev_loader.var_get('user').upper() + " Benchmark " + str(col)
                )
                config = None
                fig = go.Figure(data=data, layout=layout)
                save_path = "/REP_EXECUTION_PLANS_" + str(ev_loader.var_get('user')) + '_' + str(gathered_stats) + '_' + col + '.html'
                plot(fig, config=config, filename=self.__save_path + save_path, auto_open=False)
        #
        self.__logger.log('Report generation complete.')