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
    #
    def __explain_plan_for(self, p_sql):
        return "explain plan for " + str(p_sql)
    #
    def __query_plan_table(self):
        """
        Ensures that latest explain plan is returned from Oracle's plan_table
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
    def generateXPlan(self, sql, binds=None):
        #
        v_sql = self.__explain_plan_for(sql)
        #
        self.__db_conn.execute_dml(dml=v_sql, params=binds)
        #
        result_set = self.__db_conn.execute_query(query=self.__query_plan_table())
        #
        return result_set
#
