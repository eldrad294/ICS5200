declare
  sid number;
  serial number;
begin
  loop
	begin
	  select sid, serial#
	  into sid, serial
	  from v$session
	  where program like 'python3%'
	  and schemaname like 'TPC%'
	  and username like 'TPC%'
	  and status = 'ACTIVE'
	  and last_call_et > 60;
	  dbms_system.set_ev(sid, serial, 10237, 1, '');
	  dbms_output.put_line('Terminated SID ['||sid||'] SERIAL ['||serial||']');
	  dbms_lock.sleep(10);
	  dbms_system.set_ev(sid, serial, 10237, 0, '');
    exception
      when no_data_found then
        dbms_lock.sleep(10);
    end;
  end loop;
end;