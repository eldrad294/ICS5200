select *
from REP_EXECUTION_PLANS;
--
select 'alter system kill session '''||sid||','||serial#||'''' as dml
					from v$session
					where username like 'TPC%'
					and status = 'ACTIVE'
					and program like '%python%'
					and sysdate - NUMTODSINTERVAL('60', 'SECOND') > logon_time;
--
alter system kill session '222,40511';
