declare
  l_job number := 0;
begin
  dbms_job.submit(
    l_job,
    'begin kill_long_running(&1); end;',
    sysdate,
    null
  );
  commit;
  dbms_lock.sleep(5);
end;