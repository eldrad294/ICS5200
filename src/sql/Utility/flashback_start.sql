shutdown immediate;
startup mount;
alter database archivelog;
alter database flashback on;
FLASHBACK DATABASE TO RESTORE POINT &1;
ALTER DATABASE OPEN RESETLOGS;