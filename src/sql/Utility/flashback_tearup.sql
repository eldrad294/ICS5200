shutdown immediate;
startup mount;
alter database archivelog;
alter database flashback on;
CREATE RESTORE POINT &1 GUARANTEE FLASHBACK DATABASE;
alter database open;