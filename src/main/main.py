#
# Module Imports
import sys
from os.path import dirname, abspath
print(dirname(dirname(abspath(__file__))))
sys.path.append(dirname(dirname(abspath(__file__))))
for p in sys.path:
    print(p)
print('Entry0')
from src.utils.db_interface import db_conn
#
# Connects to Database
print('Entry1')
db_conn.connect()
print('Entry2')
