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
from src.utils.env_var_loader import ev_loader
#
# Loading of program variables
ev_loader.var_load({"project_dir":project_dir,"src_dir":src_dir})
from src.utils.logger import logger
from src.utils.db_interface import db_conn
#
# Connects to Database
db_conn.connect()
rec_cur = db_conn.query('select 1 from dual;')
print(rec_cur)
