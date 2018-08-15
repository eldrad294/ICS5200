#
# Module Imports
import hashlib
#
class XPlan:
    """
    This class serves as an interface to Oracle's explain plan generation utility, providing wrapper methods so as to
    invoke oracle explain plan generation packages, and return data in a formatted, cleaned manner.
    """
    def __init__(self, db_conn, logger, ev_loader):
        self.__db_conn = db_conn
        self.__logger = logger
        self.__ev_loader = ev_loader
        self.__execution_plan_hint = "/*ICS5200_MONITOR_HINT*/"
        self.__report_execution_plan = 'REP_EXECUTION_PLANS'
        #
        # Create reporting table
        self.__create_REP_EXECUTION_PLANS()
    #
    def __explain_plan_syntax(self, p_sql):
        return "explain plan for " + str(p_sql)
    #
    def __execution_plan_syntax(self, p_sql):
        p_sql = p_sql.split()
        r_sql = ""
        for i, sql in enumerate(p_sql):
            if i == 1:
                r_sql += " " + self.__execution_plan_hint
            r_sql += " " + sql
        return r_sql
    #
    def __query_explain_plan(self):
        """
        Ensures that latest explain plan is returned from Oracle's plan_table. The query returns the latest generated
        explain plan found in the 'plan_table'. This is risky for rapid-interweaving queries whose explain plan is
        generated atop each other.

        The above highlighted risk is minimal, since 'explain plan for' statements should not be generated by the Oracle
        instance, and therefore are only utilized through user intervention.
        :return:
        """
        return "select * " \
               "from plan_table " \
                "where plan_id = ( " \
                " select max(plan_id) " \
                " from plan_table " \
                " where to_date(to_char(timestamp,'MM/DD/YYYY'),'MM/DD/YYYY') = to_date(to_char(sysdate,'MM/DD/YYYY'),'MM/DD/YYYY') " \
                ") " \
                "order by id"
    #
    def __query_execution_plan(self, transaction_name=None, md5_sum=None):
        """
        Ensures that latest execution plan metrics are returned from Oracle's v$sqlarea view, distinguished by latest
        hint found within the view. The query identifies queries using a hint (SQL Comment), and retrieves the most
        latest one from the view.
        :param transaction_name - Name to store inside reporting table for specific transaction
        :param md5_sum - Md5 Hash sum of sql
        :return:
        """
        if transaction_name is not None:
            if md5_sum is not None:
                return "insert into " + self.__report_execution_plan + " " \
                       "select * " \
                       "from( " \
                       "select vs.*, '" + transaction_name + "', '" + md5_sum + "' " \
                       "from v$sql vs " \
                       "where sql_text like '%" + self.__execution_plan_hint + "%' " \
                       "and sql_text not like '%v_sql%' " \
                       "and sql_text not like '%V_SQL%' " \
                       "order by last_active_time desc " \
                       ") where rownum <= 1"
            else:
                raise ValueError("md5_sum was not specified!")
        else:
            return  "select * " \
                    "from( " \
                    "select * " \
                    "from v$sql " \
                    "where sql_text like '%" + self.__execution_plan_hint + "%' " \
                    "and sql_text not like '%v_sql%' " \
                    "and sql_text not like '%V_SQL%' " \
                    "order by last_active_time desc " \
                    ") where rownum <= 1"
    #
    def __select_relevant_columns(self, plan, schema, selection):
        """
        This private function iterates over full explain/execution plan, and returns only relevant columns dictated by
        selection list
        :param plan: The original, full explain/execution plan
        :param schema: The explain/execution column names passed as a list
        :param selection: The fields to be filtered according to the selection list
        :return: Returns a filtered version of the explain/execution plan + schema
        """
        #
        temp_plan, return_plan = dict(), dict()
        #
        # Converts plan into dictionary of columns
        for i in range(len(schema)):
            temp_plan[schema[i]] = []
            for row in plan:
                temp_plan[schema[i]].append(row[i])
        #
        if selection is not None:
            for sel in selection:
                if sel in temp_plan:
                    return_plan[sel] = temp_plan[sel]
            return return_plan
        else:
            return temp_plan
    #
    def __create_REP_EXECUTION_PLANS(self):
        """
        Creates reporting table REP_EXECUTION_PLANS to save v$sql execution metrics.
        The table is a replica of v$sql, with two additional columns:
        1) TPC_TRANSACTION_NAME - Contains name of TPC Transaction
        2) STATEMENT_HASH_SUM - Contains md5 hash sum for a particular statement
        :return:
        """
        sql_statement = "select count(*) from dba_tables where table_name = '" + self.__report_execution_plan + "'"
        result = int(self.__db_conn.execute_query(query=sql_statement, fetch_single=True)[0])
        if result == 0:
            if self.__ev_loader == 'True':
                dml_statement = "drop table " + self.__report_execution_plan
                self.__db_conn.execute_dml(dml=dml_statement)
                self.__logger.log('Dropped table ' + self.__report_execution_plan + " for cleanup..")
            #
            # Creates Reporting Table
            self.__logger.log('Creating table [' + self.__report_execution_plan + ']..')
            dml_statement = "create table " + self.__report_execution_plan + " tablespace users as " \
                                                                             "select * from v$sql where 1=0"
            self.__db_conn.execute_dml(dml=dml_statement)
            #
            # Adds column 'TPC_STATEMENT_NAME'
            dml_statement = "alter table " + self.__report_execution_plan + " add TPC_TRANSACTION_NAME varchar2(20)"
            self.__db_conn.execute_dml(dml=dml_statement)
            #
            # Adds column 'STATEMENT_HASH_SUM'
            dml_statement = "alter table " + self.__report_execution_plan + " add STATEMENT_HASH_SUM varchar2(4000)"
            self.__db_conn.execute_dml(dml=dml_statement)
        else:
            self.__logger.log('Table ['+self.__report_execution_plan+'] already exists..')
    #
    def check_if_plsql_block(self, statement):
        if 'begin' in statement.lower() and 'end' in statement.lower():
            return True
        return False
    #
    def generateExplainPlan(self, sql, binds=None, selection=None):
        """
        Retrieves Explain Plan - Query is not executed for explain plan retrieval
        :param sql: SQL under evaluation
        :param binds: Accepts query bind parameters as a tuple
        :param selection: Accepts list of column names which will be returned for the explain plan generation. If left
                          empty, selection is assumed to return all explain plan columns
        :return: Explain plan in dictionary format
        """
        sql = self.__explain_plan_syntax(sql)
        #
        self.__db_conn.execute_dml(dml=sql, params=binds)
        #
        plan, schema = self.__db_conn.execute_query(query=self.__query_explain_plan(), describe=True)
        #
        # Retrieves relevant columns specified in selection list
        plan = self.__select_relevant_columns(plan=plan, schema=schema, selection=selection)
        #
        return plan
    #
    def generateExecutionPlan(self, sql, binds=None, selection=None, transaction_name=None):
        """
        Retrieves Execution Plan - Query is executed for execution plan retrieval
        :param sql: SQL under evaluation
        :param binds: Accepts query bind parameters as a tuple
        :param selection: Accepts list of column names which will be returned for the explain plan generation. If left
                          empty, selection is assumed to return all execution plan columns
        :param transaction_name: Default set to None. If not specified (None), execution plan is returned to driver.
                                 Otherwise, the execution plan is saved to disk, in addition to the transaction name
                                 insie of a report table.
        :return: Execution plan in dictionary format
        """
        sql = self.__execution_plan_syntax(sql)
        if transaction_name is None:
            sql_md5 = None
        else:
            sql_md5 = hashlib.md5(sql.encode('utf-8')).hexdigest()
        #
        self.__db_conn.execute_dml(dml=sql, params=binds)
        #
        if transaction_name is not None:
            self.__db_conn.execute_dml(dml=self.__query_execution_plan(transaction_name=transaction_name, md5_sum=sql_md5))
            self.__db_conn.commit()
        else:
            plan, schema = self.__db_conn.execute_query(query=self.__query_execution_plan(transaction_name=False, md5_sum=sql_md5),
                                                        describe=True)
            #
            # Retrieves relavent columns specified in selection list
            plan = self.__select_relevant_columns(plan=plan, schema=schema, selection=selection)
            #
            return plan
"""
EXAMPLE
----------------------------------------
#
# Establishes database connection
db_conn.connect()
#
xp = XPlan(db_conn=db_conn)
v_query = "select * " \
          "from CATALOG_SALES "\
          "where cs_sold_date_sk = '2450816' "\
          "order by cs_sold_time_sk"
plan = xp.generateExplainPlan(sql=v_query, selection=['COST','DEPTH','CARDINALITY'])
print(plan)
plan = xp.generateExecutionPlan(sql=v_query, selection=['COST','DEPTH','CARDINALITY'])
print(plan)
"""
