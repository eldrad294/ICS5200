#
# Module Imports
import sys
from os.path import dirname, abspath
from src.utils.config_parser import g_config
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
from src.datageneration.tpc_generation import TPC_Wrapper
from src.utils.db_interface import db_conn
#
# Establishes database connection
db_conn.connect()
#
tpcds_generation_bool, tpce_generation_bool = bool(g_config.get_value('DataGeneration','tpcds_generation')), \
                                              bool(g_config.get_value('DataGeneration','tpce_generation'))
parallel_degree, data_size = int(g_config.get_value('DataGeneration','tpcds_generation')), \
                             int(g_config.get_value('DataGeneration','tpcds_generation'))
if tpcds_generation_bool is True:
    TPC_Wrapper.generate_data(tpc_type='TPC-DS',
                              data_size=data_size,
                              parallel_degree=parallel_degree)
elif tpce_generation_bool is True:
    raise NotImplementedError("This logic is not yet implemented!")
#
logger.log("Script Complete!\n-------------------------------------")
