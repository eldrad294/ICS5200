create or replace procedure kill_long_running
  (i_secs number)
as
  i_count number;
begin
  select count(*)
  into i_count
  from user_tables
  where table_name = 'MON_KILL_LONG_RUNNING';
  if i_count < 1 then
    execute immediate 'create table MON_KILL_LONG_RUNNING (running number default 1)';
  end if;
  while (true)
  loop
    execute immediate 'select running into i_count from MON_KILL_LONG_RUNNING' into i_count;
    if i_count = 0 then
      exit;
    end if;
    begin
      for rec in (select 'alter system kill session '''||sid||','||serial#||'''' as dml
					from v$session
					where username like 'TPC%'
					and status = 'ACTIVE'
					and program like '%python%'
					and sysdate - NUMTODSINTERVAL(i_secs, 'MINUTE') > logon_time)
      loop
        begin
          execute immediate rec.dml;
        exception
          when others then
            null;
        end;
      end loop;
    exception
      when no_data_found then
        null;
    end;
    dbms_lock.sleep(10);
  end loop;
end;
