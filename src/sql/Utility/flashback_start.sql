shutdown immediate;
startup mount;
FLASHBACK DATABASE TO RESTORE POINT &1;
alter database flashback off;
DROP RESTORE POINT &1;
alter database noarchivelog;
alter database open;