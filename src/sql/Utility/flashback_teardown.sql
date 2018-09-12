shutdown immediate;
startup mount;
DROP RESTORE POINT &1;
alter database archivelog;
alter database flashback off;
alter database open;