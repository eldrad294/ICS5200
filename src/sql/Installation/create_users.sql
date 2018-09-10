/*
TO EXECUTE UNDER SYS USER!
*/
--
-- tpcds1
grant execute on dbms_lock to tpcds1;
grant execute on tpcds1.kill_long_running to tpcds1;
grant alter system to tpcds1;
GRANT SELECT ON V_$SESSION TO tpcds1;
GRANT CREATE SESSION TO tpcds1;
GRANT ALL PRIVILEGES TO tpcds1;

--
-- tpcds10
grant execute on dbms_lock to tpcds10;
grant execute on tpcds10.kill_long_running to tpcds10;
grant alter system to tpcds10;
GRANT SELECT ON V_$SESSION TO tpcds10;
GRANT CREATE SESSION TO tpcds10;
GRANT ALL PRIVILEGES TO tpcds10;

--
-- tpcds100
grant execute on dbms_lock to tpcds100;
grant execute on tpcds100.kill_long_running to tpcds100;
grant alter system to tpcds100;
GRANT SELECT ON V_$SESSION TO tpcds100;
GRANT CREATE SESSION TO tpcds100;
GRANT ALL PRIVILEGES TO tpcds100;
