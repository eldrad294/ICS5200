shutdown immediate;
startup mount;
alter database archivelog;
alter database flashback on;
alter database open;
CREATE RESTORE POINT &1 GUARANTEE FLASHBACK DATABASE;