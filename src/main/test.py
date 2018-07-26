#
# Module Imports
import sys
from os.path import dirname, abspath
#
# Retrieving relative paths for project directory
project_dir = dirname(dirname(dirname(abspath(__file__))))
src_dir = dirname(dirname(abspath(__file__)))
#
# Appending to python path
sys.path.append(project_dir)
sys.path.append(src_dir)
#
from src.framework.env_var_loader import ev_loader
#
# Loading of program variables
ev_loader.var_load({"project_dir":project_dir,"src_dir":src_dir})
"""
--------------IMPORTANT!!--------------
-------PLACE NEW MODULES BELOW!!-------
---------------------------------------
"""
from src.framework.logger import logger
from src.data.tpc import TPC_Wrapper
from src.framework.db_interface import db_conn
from src.framework.config_parser import g_config
from src.data.loading import FileLoader
from src.framework.explain_plan_interface import XPlan
#
# Establishes database connection
db_conn.connect()
#
xp = XPlan(db_conn=db_conn)
v_query = """
select *
from CATALOG_SALES
where cs_sold_date_sk = :1
order by cs_sold_time_sk
"""
v_params = ('2450816')
print(xp.generateXPlan(v_query,v_params))