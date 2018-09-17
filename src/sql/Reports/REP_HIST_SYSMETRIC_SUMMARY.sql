create table REP_HIST_SYSMETRIC_SUMMARY as
select dhss.*,
       dhsnap.startup_time,
       dhsnap.flush_elapsed,
       dhsnap.snap_level,
       dhsnap.error_count,
       dhsnap.snap_flag,
       dhsnap.snap_timezone
from DBA_HIST_SYSMETRIC_SUMMARY dhss,
     dba_hist_snapshot dhsnap
where dhss.snap_id = dhsnap.snap_id
and dhss.dbid = dhsnap.dbid
and dhss.instance_number = dhsnap.instance_number
and 1=0;
--
select dhss.*,
       dhsnap.startup_time,
       dhsnap.flush_elapsed,
       dhsnap.snap_level,
       dhsnap.error_count,
       dhsnap.snap_flag,
       dhsnap.snap_timezone
from DBA_HIST_SYSMETRIC_SUMMARY dhss,
     dba_hist_snapshot dhsnap
where dhss.snap_id = dhsnap.snap_id
and dhss.dbid = dhsnap.dbid
and dhss.instance_number = dhsnap.instance_number
and dhsnap.snap_id between '544' and '545';