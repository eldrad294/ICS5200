#
# Module Imports
print('Entry0')
from src.utils.db_interface import db_conn
#
# Connects to Database
print('Entry1')
db_conn.connect()
print('Entry2')
