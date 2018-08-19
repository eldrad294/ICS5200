"""
---------------------------------------------------
SCRIPT WARM UP - Module Import & Path Configuration
---------------------------------------------------
"""
#
# Module Imports
import sys
from os.path import dirname, abspath
#
# Retrieving relative paths for project directory
home_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
project_dir = dirname(dirname(dirname(abspath(__file__))))
src_dir = dirname(dirname(abspath(__file__)))
#
# Appending to python path
sys.path.append(home_dir)
sys.path.append(project_dir)
sys.path.append(src_dir)
#
from src.framework.script_initializer import ScriptInitializer
from src.framework.db_interface import ConnectionPool
si = ScriptInitializer(project_dir=project_dir, src_dir=src_dir, home_dir=home_dir)
ev_loader = si.get_global_config()
db_conn = ConnectionPool.claim_from_pool()[2]
spark_context = si.initialize_spark().get_spark_context()
logger = si.initialize_logger()

from src.data.tpc import TPC_Wrapper, FileLoader
from src.framework.db_interface import DatabaseInterface
#
# TPC Wrapper Initialization
tpc = TPC_Wrapper(ev_loader=ev_loader,
                  logger=logger,
                  database_context=db_conn)
#
db_conn = DatabaseInterface(instance_name=ev_loader.var_get('instance_name'),
                                 user=ev_loader.var_get('user'),
                                 host=ev_loader.var_get('host'),
                                 service=ev_loader.var_get('service'),
                                 port=ev_loader.var_get('port'),
                                 password=ev_loader.var_get('password'),
                                 logger=logger)
db_conn.connect()
#
def __parse_data_line(dataline):
    """
    Iterates over input data line, and parses value into a list. Values are delimeted according to config file,
    default to '|'
    :param line:
    :return:
    """
    list_line = []
    delimeter = '|'
    value = ""
    for i in dataline:
        if i != delimeter:
            value += i
        else:
            try:
                if Decimal(value) % 1 == 0:
                    list_line.append(int(value))
                else:
                    list_line.append(float(value))
            except Exception:
                list_line.append(str(value))
            value = ""
    return list_line
#
# Retrieve columns required for batch insert
sql = "select column_name from user_tab_columns where table_name = '" + table_name.upper() + "' order by column_id";
res = di.execute_query(query=sql, describe=False)
column_names = "("
for i, item in enumerate(res):
    if i == 0:
        column_names += str(item[0])
    else:
        column_names += ',' + str(item[0])
else:
    column_names += ')'
#
# Iterate over RDD partition
row_count = 0
values_bank = []
dml = "INSERT INTO " + table_name + " " + column_names + " VALUES ("
for count, data_line in enumerate(['2450815|1|1|211|','2450815|2|1|235|','2450815|4|1|859|']):
    l_line = __parse_data_line(dataline=data_line)
    if count < 1:
        for i in range(len(l_line)):
            if i == 0:
                dml += ":" + str(i+1)
            else:
                dml += ",:" + str(i+1)
        dml += ")"
    values_bank.append(l_line)
    row_count += 1
    if count % 1000 == 0 and count != 0:
        di.execute_many_dml(dml=dml, data=values_bank)  # Bulk Insert
        di.commit()  # Commit once after every RDD batch
        logger.log('Committed 1000 batch for [' + table_name + ']')
        values_bank = []
#
# Execute remaining rows
di.execute_many_dml(dml=dml, data=values_bank)  # Bulk Insert
di.commit()  # Commit once after every RDD batch
di.close()