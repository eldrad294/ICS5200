shutdown immediate;
startup mount;
DROP RESTORE POINT &1;
alter database flashback off;
alter database noarchivelog;
alter database open;