/*
Create Tablespaces

* Tablespace | Datafiles can be checked from:
select * from dba_tablespaces;
select * from DBA_DATA_FILES;

NB: Ensure that physical datafiles are deleted before running this script:
1) CISK> cd /mnt/raid5/oradata/gabsam/
2) CISK> sudo rm -f tpcds*
*/
CREATE TABLESPACE tpcds1   DATAFILE '/mnt/raid5/oradata/gabsam/tpcds1_01.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
CREATE TABLESPACE tpcds10  DATAFILE '/mnt/raid5/oradata/gabsam/tpcds10_01.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
CREATE TABLESPACE tpcds50  DATAFILE '/mnt/raid5/oradata/gabsam/tpcds50_01.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
CREATE TABLESPACE tpcds100 DATAFILE '/mnt/raid5/oradata/gabsam/tpcds100_01.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
/*
Create datafiles
*/
ALTER TABLESPACE tpcds1   ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds1_02.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds1   ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds1_03.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds10  ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds10_02.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds10  ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds10_03.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds50  ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds50_02.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds50  ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds50_03.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds100 ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds100_02.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;
ALTER TABLESPACE tpcds100 ADD DATAFILE '/mnt/raid5/oradata/gabsam/tpcds100_03.dbf' SIZE 5M AUTOEXTEND ON NEXT 1024K MAXSIZE UNLIMITED;