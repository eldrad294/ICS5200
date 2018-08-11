from src.reports.bar import BarCharts
from src.framework.db_interface import ConnectionPool
connection_details = {'instance_name':'gabsam',
                              'user':'tpcds1',
                              'host':'192.168.202.222',
                              'service':'gabsam',
                              'port':'1521',
                              'password':'tpc'}
ConnectionPool.create_connection_pool(max_connections=1,
                                      connection_details=connection_details,
                                      logger=logger)
#
bc = BarCharts(ConnectionPool.claim_from_pool())
bc.generate_REP_TPC_DESCRIBE()