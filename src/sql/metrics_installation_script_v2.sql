create or replace procedure PERF_EXTRACT_DATA
is
  i_count     pls_integer                      := 0;
  v_retention varchar2(1)                      := 1;   							  -- number of days to retrieve data from
  v_parallel_degree varchar2(2)                := 10;
  --
  v_MSC_DEBUG varchar2(400)                    := 'MSC_DEBUG';                      -- debug table for logging purposes
  v_MSC_VSQL varchar2(200)                     := 'MSC_VSQL';        		        -- v$sql
  v_MSC_VSQLAREA varchar2(200)                 := 'MSC_VSQLAREA';			        -- v$sqlarea
  v_MSC_SQLWORKAREA varchar2(200)              := 'MSC_VSQL_WORKAREA';              -- v$sql_workarea
  v_MSC_ASH varchar2(200)                      := 'MSC_ASH';                        -- v$active_session_history
  v_MSC_RMAN_BJD varchar2(200)                 := 'MSC_RMAN_BJD';                   -- v$rman_backup_job_details
  v_MSC_SQL_PLAN_STATS varchar2(200)           := 'MSC_SQL_PLAN_STATS';             -- v$sql_plan_statistics
  v_MSC_SQL_PLAN_STATS_ALL varchar2(200)       := 'MSC_SQL_PLAN_STATS_ALL';         -- v$sql_plan_statistics_all
  v_MSC_SQLSTATS varchar2(200)                 := 'MSC_SQLSTATS';			        -- v$sqlstats
  v_MSC_DBA_SJL varchar2(200)                  := 'MSC_DBA_SJL';                    -- dba_scheduler_job_log
  v_MSC_DBA_HIST_SNAPSHOT varchar2(200)        := 'MSC_DBA_HIST_SNAPSHOT';          -- dba_hist_snapshot
  v_MSC_DBA_HIST_ASH varchar2(200)             := 'MSC_DBA_HIST_ASH';		        -- dba_hist_active_sess_history
  v_MSC_SQL_BIND_CAPTURE varchar2(200)         := 'MSC_SQL_BIND_CAPTURE';           -- v$sql_bind_capture
--v_MSC_DBA_TABLES varchar2(200)               := 'MSC_DBA_TABLES';                 -- dba_tables
--v_MSC_DBA_INDEXES varchar2(200)              := 'MSC_DBA_INDEXES';                -- dba_indexes
  v_MSC_DBA_AWR_HIST varchar2(4000)            := 'MSC_DBA_AWR_HIST';               -- AWR Snapshot Extraction Table
  v_MSC_DBA_SQL_HIST varchar2(200)             := 'MSC_DBA_SQL_HIST';               -- dba_hist_snapshot, dba_hist_sqltext, dba_hist_sqlstat
  v_MSC_DBA_HIST_SYSMETRIC varchar2(200)       := 'MSC_DBA_HIST_SYSMETRIC';         -- dba_hist_sysmetric
  v_MSC_DBA_HIST_SYSSTAT varchar2(200)         := 'MSC_DBA_HIST_SYSSTAT';           -- dba_hist_sysstat
  v_MSC_DBA_HIST_SYSTEM_EVENT varchar2(200)    := 'MSC_DBA_HIST_SYSTEM_EVENT';      -- dba_hist_system_event
  v_MSC_DBA_HIST_IOSTAT_DETAIL varchar2(200)   := 'MSC_DBA_HIST_IOSTAT_DETAIL';     -- dba_hist_iostat_detail
  v_MSC_DBA_HIST_EVENT_HISTOGRAM varchar2(200) := 'MSC_DBA_HIST_EVENT_HISTOGRAM';   -- dba_hist_event_histogram
  v_MSC_DBA_HIST_SERVICE_STAT varchar2(200)    := 'MSC_DBA_HIST_SERVICE_STAT';      -- dba_hist_service_stat
begin
  --
  -- MSC_DEBUG
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DEBUG;
    --
    if i_count = 0 then
      --
      execute immediate 'create table MSC_DEBUG(
						   message varchar2(800) not null,
						   log_date varchar2(30) not null
				         ) tablespace USERS';
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values (''Created Debug Table'', '''||to_char(sysdate)||''')';
      commit;
    end if;
    --dbms_output.put_line(v_MSC_VSQL||' extract complete..');
  exception
    when others then
      rollback;
      dbms_output.put_line(sqlerrm);
      commit;
  end;
  --
  -- v$sql
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_VSQL;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_VSQL||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from v$sql
                           where last_active_time is not null
                           order by last_active_time';
    else
      --
      execute immediate 'insert into '||v_MSC_VSQL||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from v$sql
		                   where last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   last_active_time is not null
		                   order by last_active_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_VSQL||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$sqlarea
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_VSQLAREA;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_VSQLAREA||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from v$sqlarea
                           where last_active_time is not null
                           order by last_active_time';
    else
      --
      execute immediate 'insert into '||v_MSC_VSQLAREA||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from v$sqlarea
		                   where last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   last_active_time is not null
		                   order by last_active_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_VSQLAREA||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$sql_workarea
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_SQLWORKAREA;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_SQLWORKAREA||' tablespace USERS as
                           select /*+PARALLEL('||v_parallel_degree||')*/ sw.*
						   from   v$sql_workarea sw,
       							  v$sql s
						   where  s.sql_id = sw.sql_id
						   and    s.last_active_time is not null
						   order by s.last_active_time,
	     							sw.operation_id,
	     							sw.child_number';
    else
      --
      execute immediate 'insert into '||v_MSC_SQLWORKAREA||'
                           select /*+PARALLEL('||v_parallel_degree||')*/ sw.*
						   from   v$sql_workarea sw,
       							  v$sql s
						   where  s.sql_id = sw.sql_id
						   and    s.last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
  				 		   and    to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
  				 		   and    s.last_active_time is not null
						   order by s.last_active_time,
	     							sw.operation_id,
	     							sw.child_number';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_SQLWORKAREA||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$active_session_history
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_ASH;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_ASH||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from v$active_session_history
                           where sample_time is not null
                           order by sample_time';
    else
      --
      execute immediate 'insert into '||v_MSC_ASH||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from v$active_session_history
		                   where sample_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   sample_time is not null
		                   order by sample_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_ASH||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$rman_backup_job_details
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_RMAN_BJD;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_RMAN_BJD||' tablespace USERS as
                           select  /*PARALLEL('||v_parallel_degree||')*/ *
                           from v$rman_backup_job_details
                           where start_time is not null
                           order by start_time';
    else
      --
      execute immediate 'insert into '||v_MSC_RMAN_BJD||'
                           select /*PARALLEL('||v_parallel_degree||')*/ *
		                   from v$rman_backup_job_details
		                   where start_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   start_time is not null
		                   order by start_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_RMAN_BJD||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$sql_plan_statistics
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_SQL_PLAN_STATS;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_SQL_PLAN_STATS||' tablespace USERS as
                           select /*+PARALLEL('||v_parallel_degree||')*/sps.*
						   from   v$sql_plan_statistics sps,
                                  v$sql s
                           where  s.sql_id = sps.sql_id
                           and    s.last_active_time is not null
                           order by s.last_active_time,
                                    sps.operation_id,
                                    sps.child_number';
    else
      --
      execute immediate 'insert into '||v_MSC_SQL_PLAN_STATS||'
                           select /*+PARALLEL('||v_parallel_degree||')*/sps.*
						   from   v$sql_plan_statistics sps,
                                  v$sql s
                           where  s.sql_id = sps.sql_id
                           and    s.last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
                           and    to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
                           and    s.last_active_time is not null
                           order by s.last_active_time,
                                    sps.operation_id,
                                    sps.child_number';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_SQL_PLAN_STATS||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$sql_plan_statistics_all
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_SQL_PLAN_STATS_ALL;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_SQL_PLAN_STATS_ALL||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from v$sql_plan_statistics_all
                           where timestamp is not null
                           order by timestamp, id';
    else
      --
      execute immediate 'insert into '||v_MSC_SQL_PLAN_STATS_ALL||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from v$sql_plan_statistics_all
		                   where timestamp between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   timestamp is not null
		                   order by timestamp, id';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_SQL_PLAN_STATS_ALL||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$sqlstats
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_SQLSTATS;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_SQLSTATS||' tablespace USERS as
                           select /*+PARALLEL('||v_parallel_degree||')*/ *
						   from v$sqlstats
						   where last_active_time is not null
						   order by last_active_time';
    else
      --
      execute immediate 'insert into '||v_MSC_SQLSTATS||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from v$sqlstats
		                   where last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   last_active_time is not null
		                   order by last_active_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_SQLSTATS||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_scheduler_job_log
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_SJL;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_SJL||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from dba_scheduler_job_log
                           where log_date is not null
                           order by log_date';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_SJL||'
                           select /*+PARALLEL('||v_parallel_degree||')*/ *
		                   from dba_scheduler_job_log
		                   where log_date between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and   log_date is not null
		                   order by log_date';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_SJL||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_snapshot
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_SNAPSHOT;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_SNAPSHOT||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from dba_hist_snapshot
                           where begin_interval_time is not null
                           order by begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_SNAPSHOT||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from dba_hist_snapshot
		                   where begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and  to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and  begin_interval_time is not null
		                   order by begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_SNAPSHOT||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_active_sess_history
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_ASH;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_ASH||' tablespace USERS as
                           select  /*+PARALLEL('||v_parallel_degree||')*/ *
                           from dba_hist_active_sess_history
                           where sample_time is not null
                           order by sample_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_ASH||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
		                   from dba_hist_active_sess_history
		                   where sample_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
		                   and  to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
		                   and  sample_time is not null
		                   order by sample_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_ASH||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- v$sql_bind_capture
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_SQL_BIND_CAPTURE;
    --
    -- Retrieve everything from v$sql_bind_capture except for VALUE_STRING & VALUE_ANYDATA for security purposes
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_SQL_BIND_CAPTURE||' tablespace USERS  as
					       select /*+PARALLEL('||v_parallel_degree||')*/sbc.address, sbc.hash_value, sbc.sql_id, sbc.child_address, sbc.child_number, sbc.name, sbc.position, sbc.dup_position, sbc.datatype, sbc.datatype_string, sbc.character_sid, sbc.precision, sbc.scale, sbc.max_length, sbc.was_captured, sbc.last_captured
						   from   v$sql_bind_capture sbc,
					              v$sql s
						   where  s.sql_id = sbc.sql_id
						   and    s.last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and    to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
						   and    s.last_active_time is not null
						   order by s.last_active_time,
							        sbc.child_number';

    else
      --
      execute immediate 'insert into '||v_MSC_SQL_BIND_CAPTURE||'
                           select /*+PARALLEL('||v_parallel_degree||')*/sbc.address, sbc.hash_value, sbc.sql_id, sbc.child_address, sbc.child_number, sbc.name, sbc.position, sbc.dup_position, sbc.datatype, sbc.datatype_string, sbc.character_sid, sbc.precision, sbc.scale, sbc.max_length, sbc.was_captured, sbc.last_captured, sbc.con_id
						   from   v$sql_bind_capture sbc,
					              v$sql s
						   where  s.sql_id = sbc.sql_id
						   and    s.last_active_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and    to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
						   and    s.last_active_time is not null
						   order by s.last_active_time,
							        sbc.child_number';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_SQL_BIND_CAPTURE||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_tables
--  begin
--    --
--    select count(*)
--    into i_count
--    from user_tables
--    where table_name = v_MSC_DBA_TABLES;
--    --
--    if i_count = 0 then
--      --
--      execute immediate 'create table '||v_MSC_DBA_TABLES||' as
--                           select  /*+ PARALLEL('||v_parallel_degree||')*/ *
--                           from dba_tables';
--    else
--      --
--      execute immediate 'insert into '||v_MSC_DBA_TABLES||'
--                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
--		                   from dba_tables';
--    end if;
--    commit;
--    dbms_output.put_line(v_MSC_DBA_TABLES||' extract complete..');
--  exception
--    when others then
--      rollback;
--      dbms_output.put_line(sqlerrm);
--  end;
--  --
--  --dba_indexes
--  begin
--    --
--    select count(*)
--    into i_count
--    from user_tables
--    where table_name = v_MSC_DBA_INDEXES;
--    --
--    if i_count = 0 then
--      --
--      execute immediate 'create table '||v_MSC_DBA_INDEXES||' as
--                           select  /*+ PARALLEL('||v_parallel_degree||')*/ *
--                           from dba_indexes';
--    else
--      --
--      execute immediate 'insert into '||v_MSC_DBA_INDEXES||'
--                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
--		                   from dba_indexes';
--    end if;
--    commit;
--    dbms_output.put_line(v_MSC_DBA_INDEXES||' extract complete..');
--  exception
--    when others then
--      rollback;
--      dbms_output.put_line(sqlerrm);
--  end;
  --
  -- AWR Snapshot Extraction Table
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_AWR_HIST;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_AWR_HIST||'(
      						output varchar2(1500),
      						snap_id varchar2(10),
      						time_stamp varchar2(30)) tablespace USERS ';
    end if;
    --
    -- Extraction of awr (HTML format and dumping into temporary table, along with respective snap id and timestamp)
    for rec in (select snap_id, dbid
    			from dba_hist_snapshot
    			where begin_interval_time
    			between to_date(to_char(sysdate-1,'DD/MM/YYYY HH24:MI:SS'), 'DD/MM/YYYY HH24:MI:SS')
		        and to_date(to_char(sysdate,'DD/MM/YYYY HH24:MI:SS'), 'DD/MM/YYYY HH24:MI:SS')
		        order by begin_interval_time)
    loop
      begin
        --
        execute immediate 'insert into '||v_MSC_DBA_AWR_HIST||' (output, snap_id, time_stamp)
        					 select output, '||to_char(rec.snap_id)||', sysdate
        					 from table(dbms_workload_repository.awr_report_html('||to_char(rec.dbid)||',1,'||to_char(rec.snap_id)||','||to_char(rec.snap_id + 1)||'))';
        execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_AWR_HIST||' extract complete'', '''||to_char(sysdate)||''')';
        commit;
      exception
        when others then
          rollback;
          execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
          commit;
      end;
    end loop;
    --
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_AWR_HIST||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_snapshot
  -- dba_hist_sqltext
  -- dba_hist_sqlstat
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_SQL_HIST;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_SQL_HIST||' tablespace USERS as
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhst.sql_text,dhss.*
						   from dba_hist_snapshot dhs,
                                dba_hist_sqltext dhst,
                                dba_hist_sqlstat dhss
                           where dhs.snap_id = dhss.snap_id
                           and   dhss.sql_id = dhst.sql_id
                           and   begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
                           and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
                           order by begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_SQL_HIST||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhst.sql_text,dhss.*
						   from dba_hist_snapshot dhs,
                                dba_hist_sqltext dhst,
                                dba_hist_sqlstat dhss
                           where dhs.snap_id = dhss.snap_id
                           and   dhss.sql_id = dhst.sql_id
                           and   begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
                           and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
                           order by begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_SQL_HIST||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_sysmetric
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_SYSMETRIC;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_SYSMETRIC||' tablespace USERS as
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
						   from dba_hist_sysmetric_summary
						   where 1=1
						   and   begin_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
						   order by snap_id, begin_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_SYSMETRIC||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ *
						   from dba_hist_sysmetric_summary
						   where 1=1
						   and   begin_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and   to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'') --Returns n day snapshot
						   order by snap_id, begin_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_SYSMETRIC||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_sysstat
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_SYSSTAT;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_SYSSTAT||' tablespace USERS as
	 					 select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						 from dba_hist_sysstat dhs,
						      dba_hist_snapshot dhss
						 where dhs.snap_id = dhss.snap_id
						 and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 order by dhss.snap_id, dhss.begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_SYSSTAT||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						   from dba_hist_sysstat dhs,
						        dba_hist_snapshot dhss
						   where dhs.snap_id = dhss.snap_id
						   and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   order by dhss.snap_id, dhss.begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_SYSSTAT||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_system_event
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_SYSTEM_EVENT;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_SYSTEM_EVENT||' tablespace USERS as
	 					 select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						 from dba_hist_system_event dhs,
						      dba_hist_snapshot dhss
						 where dhs.snap_id = dhss.snap_id
						 and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 order by dhss.snap_id, dhss.begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_SYSTEM_EVENT||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						   from dba_hist_system_event dhs,
						        dba_hist_snapshot dhss
						   where dhs.snap_id = dhss.snap_id
						   and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   order by dhss.snap_id, dhss.begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_SYSTEM_EVENT||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_iostat_detail
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_IOSTAT_DETAIL;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_IOSTAT_DETAIL||' tablespace USERS as
	 					 select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						 from dba_hist_iostat_detail dhs,
						      dba_hist_snapshot dhss
						 where dhs.snap_id = dhss.snap_id
						 and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 order by dhss.snap_id, dhss.begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_IOSTAT_DETAIL||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						   from dba_hist_iostat_detail dhs,
						        dba_hist_snapshot dhss
						   where dhs.snap_id = dhss.snap_id
						   and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   order by dhss.snap_id, dhss.begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_IOSTAT_DETAIL||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_event_histogram
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_EVENT_HISTOGRAM;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_EVENT_HISTOGRAM||' tablespace USERS as
	 					 select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						 from dba_hist_iostat_detail dhs,
						      dba_hist_snapshot dhss
						 where dhs.snap_id = dhss.snap_id
						 and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 order by dhss.snap_id, dhss.begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_EVENT_HISTOGRAM||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						   from dba_hist_iostat_detail dhs,
						        dba_hist_snapshot dhss
						   where dhs.snap_id = dhss.snap_id
						   and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   order by dhss.snap_id, dhss.begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_EVENT_HISTOGRAM||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  -- dba_hist_service_stat
  begin
    --
    select count(*)
    into i_count
    from user_tables
    where table_name = v_MSC_DBA_HIST_SERVICE_STAT;
    --
    if i_count = 0 then
      --
      execute immediate 'create table '||v_MSC_DBA_HIST_SERVICE_STAT||' tablespace USERS as
	 					 select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						 from dba_hist_service_stat dhs,
						      dba_hist_snapshot dhss
						 where dhs.snap_id = dhss.snap_id
						 and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						 order by dhss.snap_id, dhss.begin_interval_time';
    else
      --
      execute immediate 'insert into '||v_MSC_DBA_HIST_SERVICE_STAT||'
                           select /*+ PARALLEL('||v_parallel_degree||')*/ dhs.*, dhss.begin_interval_time
						   from dba_hist_service_stat dhs,
						        dba_hist_snapshot dhss
						   where dhs.snap_id = dhss.snap_id
						   and begin_interval_time between to_date(to_char(sysdate-'||v_retention||',''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   and to_date(to_char(sysdate,''DD/MM/YYYY HH24:MI:SS''), ''DD/MM/YYYY HH24:MI:SS'')
						   order by dhss.snap_id, dhss.begin_interval_time';
    end if;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||v_MSC_DBA_HIST_SERVICE_STAT||' extract complete'', '''||to_char(sysdate)||''')';
    commit;
  exception
    when others then
      rollback;
      execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
      commit;
  end;
  --
  execute immediate 'insert into '||v_MSC_DEBUG|| ' values (''Script Complete.'', '''||to_char(sysdate)||''')';
  commit;
  --
exception
  when others then
    rollback;
    execute immediate 'insert into '||v_MSC_DEBUG|| ' values ('''||sqlerrm||''', '''||to_char(sysdate)||''')';
    commit;
end;
/
--
-- Create dba job
begin
  dbms_scheduler.create_job(
    job_name => 'PERF_JOB_EXTRACT_DATA',
    job_type => 'STORED_PROCEDURE',
    job_action  => 'PERF_EXTRACT_DATA',
    start_date => '03-JUL-2018 10.48.00 AM',
    repeat_interval => 'FREQ=DAILY;',
    end_date => '30-DEC-2019 11.59.00 PM',
    auto_drop => FALSE,
    comments => 'This is a data extraction job designed to record performance metrics'
  );
end;
/
--
-- Enable dba job
begin
  dbms_scheduler.enable('PERF_JOB_EXTRACT_DATA');
end;
/
