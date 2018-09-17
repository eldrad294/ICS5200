shutdown immediate;
startup mount;
alter database archivelog;
FLASHBACK DATABASE TO RESTORE POINT &1;
ALTER DATABASE OPEN RESETLOGS;