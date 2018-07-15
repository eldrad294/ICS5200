--
-- Dropping Schema
drop procedure PERF_EXTRACT_DATA;
--
-- Drop dba job
begin
  dbms_scheduler.drop_job(
    job_name => 'PERF_JOB_EXTRACT_DATA',
    defer => FALSE,
    commit_semantics => 'STOP_ON_FIRST_ERROR'
  );
end;
/
--
-- Dropping Tables
drop table MSC_DEBUG;
drop table MSC_SQL_BIND_CAPTURE;
drop table MSC_VSQL;
drop table MSC_VSQLAREA;
drop table MSC_VSQL_WORKAREA;
drop table MSC_ASH;
drop table MSC_RMAN_BJD;
drop table MSC_SQL_PLAN_STATS;
drop table MSC_SQL_PLAN_STATS_ALL;
drop table MSC_SQLSTATS;
drop table MSC_DBA_SJL;
drop table MSC_DBA_HIST_SNAPSHOT;
drop table MSC_DBA_HIST_ASH;
drop table MSC_SQL_BIND_CAPTURE;
drop table MSC_DBA_AWR_HIST;
drop table MSC_DBA_SQL_HIST;
drop table MSC_DBA_HIST_SYSMETRIC;
drop table MSC_DBA_HIST_SYSSTAT;
drop table MSC_DBA_HIST_SYSTEM_EVENT;
drop table MSC_DBA_HIST_IOSTAT_DETAIL;
drop table MSC_DBA_HIST_EVENT_HISTOGRAM;
drop table MSC_DBA_HIST_SERVICE_STAT;
drop table MSC_DBA_SJ_RUN_DETAILS;
