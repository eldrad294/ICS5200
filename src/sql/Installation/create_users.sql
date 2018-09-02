/*
This script creates & grants privilege to all test schemas utilized in this project
*/
--
-- tpcds1
create user tpcds1 identified by tpc;
grant connect to tpcds1;
grant connect, resource, dba to tpcds1;
grant unlimited tablespace to tpcds1;
grant execute on dbms_lock to tpcds1;
grant execute on kill_long_running to tpcds1;
grant alter system to tpcds1;
GRANT SELECT ON V_$SESSION TO tpcds1;
--
-- tpcds10
create user tpcds10 identified by tpc;
grant connect to tpcds10;
grant connect, resource, dba to tpcds10;
grant unlimited tablespace to tpcds10;
grant execute on dbms_lock to tpcds10;
grant execute on kill_long_running to tpcds10;
grant alter system to tpcds10;
GRANT SELECT ON V_$SESSION TO tpcds10;
--
-- tpcds50
create user tpcds50 identified by tpc;
grant connect to tpcds50;
grant connect, resource, dba to tpcds50;
grant unlimited tablespace to tpcds50;
grant execute on dbms_lock to tpcds50;
grant execute on kill_long_running to tpcds50;
grant alter system to tpcds50;
GRANT SELECT ON V_$SESSION TO tpcds50;
--
-- tpcds100
create user tpcds100 identified by tpc;
grant connect to tpcds100;
grant connect, resource, dba to tpcds100;
grant unlimited tablespace to tpcds100;
grant execute on dbms_lock to tpcds100;
grant execute on kill_long_running to tpcds100;
grant alter system to tpcds100;
GRANT SELECT ON V_$SESSION TO tpcds100;
--
