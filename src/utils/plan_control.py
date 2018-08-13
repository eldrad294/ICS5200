#
# Module Imports
#
class XPlan:
    """
    This class serves as an interface to Oracle's explain plan generation utility, providing wrapper methods so as to
    invoke oracle explain plan generation packages, and return data in a formatted, cleaned manner.
    """
    def __init__(self, db_conn, logger):
        self.__db_conn = db_conn
        self.__logger = logger
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
    def __query_execution_plan(self, save_to_disk=False):
        """
        Ensures that latest execution plan metrics are returned from Oracle's v$sqlarea view, distinguished by latest
        hint found within the view. The query identifies queries using a hint (SQL Comment), and retrieves the most
        latest one from the view.
        :return:
        """
        if save_to_disk:
            return "insert into " + self.__report_execution_plan + " " \
                   "select * " \
                   "from( " \
                   "select * " \
                   "from v$sql " \
                   "where sql_text like '%" + self.__execution_plan_hint + "%' " \
                   "and sql_text not like '%v_sql%' " \
                   "and sql_text not like '%V_SQL%' " \
                   "order by last_active_time desc " \
                   ") where rownum <= 1"
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
        Creates reporting table REP_EXECUTION_PLANS to save v$sql execution metrics
        :return:
        """
        sql_statement = "select count(*) from dba_tables where table_name = '" + self.__report_execution_plan + "'"
        result = int(self.__db_conn.execute_query(query=sql_statement, fetch_single=True)[0])
        if result == 0:
            dml_statement = "create table " + self.__report_execution_plan + " tablespace users as " \
                                                                             "(select * from v$sql where 1=0);"
            self.__db_conn.execute_dml(dml=dml_statement)
            self.__logger.log('Created table ['+self.__report_execution_plan+']..')
        else:
            self.__logger.log('Table ['+self.__report_execution_plan+'] already exists..')
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
    def generateExecutionPlan(self, sql, binds=None, selection=None, save_to_disk=False):
        """
        Retrieves Execution Plan - Query is executed for execution plan retrieval
        :param sql: SQL under evaluation
        :param binds: Accepts query bind parameters as a tuple
        :param selection: Accepts list of column names which will be returned for the explain plan generation. If left
                          empty, selection is assumed to return all execution plan columns
        :param save_to_disk: Default set to False. Determines whether explain plan results are saved to disk, if param
                             is set to true. Otherwise return table results.
        :return: Execution plan in dictionary format
        """
        sql = self.__execution_plan_syntax(sql)
        #
        self.__db_conn.execute_dml(dml=sql, params=binds)
        #
        if save_to_disk:
            self.__db_conn.execute_dml(dml=self.__query_execution_plan(save_to_disk=True))
            self.__db_conn.commit()
        else:
            plan, schema = self.__db_conn.execute_query(query=self.__query_execution_plan(save_to_disk=False),
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
