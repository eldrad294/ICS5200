create table REP_HIST_SNAPSHOT as
select dhsql.*,
       dhsnap.startup_time,
       dhsnap.begin_interval_time,
       dhsnap.end_interval_time,
       dhsnap.flush_elapsed,
       dhsnap.snap_level,
       dhsnap.error_count,
       dhsnap.snap_flag,
       dhsnap.snap_timezone
from dba_hist_sqlstat dhsql,
     dba_hist_snapshot dhsnap
where dhsql.snap_id = dhsnap.snap_id
and dhsql.dbid = dhsnap.dbid
and dhsql.instance_number = dhsnap.instance_number
and 1=0;
--
select dhsql.*,
       dhsnap.startup_time,
       dhsnap.begin_interval_time,
       dhsnap.end_interval_time,
       dhsnap.flush_elapsed,
       dhsnap.snap_level,
       dhsnap.error_count,
       dhsnap.snap_flag,
       dhsnap.snap_timezone,
       dhsnap.con_id
from dba_hist_sqlstat dhsql,
     dba_hist_snapshot dhsnap
where dhsql.snap_id = dhsnap.snap_id
and dhsql.dbid = dhsnap.dbid
and dhsql.instance_number = dhsnap.instance_number
and dhsnap.snap_id between '544' and '545';