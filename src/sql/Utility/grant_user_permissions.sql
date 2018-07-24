create user tpcds identified by ******;  -- Insert Password
grant connect to tpcds;
grant connect, resource, dba to tpcds;
grant create session grant any privileges to tpcds;
grant unlimited tablespace to tpcds;
