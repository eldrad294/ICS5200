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
        self.__accepted_plan_formats = ('explain','execution')
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
    def generateXPlan(self, sql, binds=None, type='explain'):
        """
        Main caller method for this class - has 2 modes of functionality depending on type parameter
        :param sql: SQL under evaluation
        :param binds: Accepts query bind parameters as a tuple
        :param type: Mode of processing - Accepts either 'explain' for explain plan generation (default), or 'execute'
                     for execution plan generation
        :return: Explain/Execution plan in tabular format
        """
        #
        plan = None
        if type == self.__accepted_plan_formats[0]:
            #
            v_sql = self.__explain_plan_for(sql)
            #
            self.__db_conn.execute_dml(dml=v_sql, params=binds)
            #
            plan = self.__db_conn.execute_query(query=self.__query_plan_table())
        elif type == self.__accepted_plan_formats[1]:
            #
            raise NotImplementedError("This logic is not yet implemented!")
        else:
            raise ValueError('Parameter type incorrect! Must be as follows: [' + self.__accepted_plan_formats + ']')
        #
        return plan
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
print(xp.generateXPlan(v_query))
"""
