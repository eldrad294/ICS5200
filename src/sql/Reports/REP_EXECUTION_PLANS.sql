create table REP_EXECUTION_PLANS tablespace users as
(select * from v$sql where 1=0);