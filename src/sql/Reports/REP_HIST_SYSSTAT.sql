create table rep_hist_sysstat as
select dhsys.*,
       dhs.startup_time,
       dhs.begin_interval_time,
       dhs.end_interval_time,
       dhs.flush_elapsed,
       dhs.snap_level,
       dhs.error_count,
       dhs.snap_flag,
       dhs.snap_timezone
from DBA_HIST_SYSSTAT dhsys,
     dba_hist_snapshot dhs
where dhsys.snap_id = dhs.snap_id
and dhsys.dbid = dhs.dbid
and dhsys.instance_number = dhs.instance_number
and 1=0;
--
select dhsys.*,
       dhs.startup_time,
       dhs.begin_interval_time,
       dhs.end_interval_time,
       dhs.flush_elapsed,
       dhs.snap_level,
       dhs.error_count,
       dhs.snap_flag,
       dhs.snap_timezone
from DBA_HIST_SYSSTAT dhsys,
     dba_hist_snapshot dhs
where dhsys.snap_id = dhs.snap_id
and dhsys.dbid = dhs.dbid
and dhsys.instance_number = dhs.instance_number
and dhs.snap_id = '618';