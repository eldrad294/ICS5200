create or replace procedure kill_long_running
  (i_secs in number := 60)
as
begin
  while (true)
  loop
    --
    begin
      for rec in (select 'alter system kill session '''||sid||','||serial#||'''' as dml
					from v$session
					where username like 'TPC%'
					and status = 'ACTIVE'
					and program like '%python%'
					and sysdate - interval ''||i_secs||'' minute > logon_time)
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
    --
    dbms_lock.sleep(10);
  end loop;
end;
