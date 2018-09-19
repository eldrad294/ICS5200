shutdown immediate;
startup mount;
alter database flashback off;
DROP RESTORE POINT &1;
alter database noarchivelog;
alter database open;