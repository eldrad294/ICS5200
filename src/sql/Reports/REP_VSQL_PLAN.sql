create table REP_VSQL_PLAN as
select *
from v$sql_plan
where sql_id in (
	select dhsql.sql_id
	from dba_hist_sqlstat dhsql,
	     dba_hist_snapshot dhsnap
	where dhsql.snap_id = dhsnap.snap_id
	and dhsql.dbid = dhsnap.dbid
	and dhsql.instance_number = dhsnap.instance_number
	and dhsnap.snap_id between '544' and '545'
)
and 1=0;
--
select *
from v$sql_plan
where sql_id in (
	select dhsql.sql_id
	from dba_hist_sqlstat dhsql,
	     dba_hist_snapshot dhsnap
	where dhsql.snap_id = dhsnap.snap_id
	and dhsql.dbid = dhsnap.dbid
	and dhsql.instance_number = dhsnap.instance_number
	and dhsnap.snap_id between '544' and '545'
);
--