#
# Module Imports
from src.framework.db_interface import db_conn
#
class XPlan:
    """
    This class serves as an interface to Oracle's explain plan generation utility, providing wrapper methods so as to
    invoke oracle explain plan generation packages, and return data in a formatted, cleaned manner.
    """
    def __init__(self, db_conn):
        self.__db_conn = db_conn
        self.__execution_plan_hint = "/*ICS5200_MONITOR_HINT*/"
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
    def __query_execution_plan(self):
        """
        Ensures that latest execution plan metrics are returned from Oracle's v$sql_plan view, distinguished by latest
        hint found within the view. The query identifies queries using a hint (SQL Comment), and retrieves the most
        latest one from the view.
        :return:
        """
        return "select * " \
               "from v$sql_plan " \
               "where sql_id = ( " \
               "  select sql_id " \
               "  from ( " \
               "    select SQL_ID " \
               "    from v$sql " \
               "    where sql_fulltext like '%ICS5200_MONITOR_HINT%' " \
               "    and sql_fulltext not like '%v$sql%' " \
               "    and sql_fulltext not like '%V$SQL%' " \
               "    order by last_active_time desc " \
               "  ) where rownum = 1 " \
               ") order by id"
    #
    def generateExplainPlan(self, sql, binds=None):
        """
        Retrieves Explain Plan - Query is not executed for explain plan retrieval
        :param sql: SQL under evaluation
        :param binds: Accepts query bind parameters as a tuple
        :return: Explain plan in tabular format
        """
        sql = self.__explain_plan_syntax(sql)
        #
        self.__db_conn.execute_dml(dml=sql, params=binds)
        #
        plan, schema = self.__db_conn.execute_query(query=self.__query_explain_plan(), describe=True)
        #
        return plan, schema
    #
    def generateExecutionPlan(self, sql, binds=None):
        """
        Retrieves Execution Plan - Query is executed for execution plan retrieval
        :param sql:
        :param binds:
        :return: Execution plan in tabular format
        """
        sql = self.__execution_plan_syntax(sql)
        #
        self.__db_conn.execute_dml(dml=sql, params=binds)
        #
        plan, schema = self.__db_conn.execute_query(query=self.__query_execution_plan(), describe=True)
        #
        return plan, schema
#
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
print(xp.generateExplainPlan(v_query))
print(xp.generateExecutionPlan(v_query))
"""
