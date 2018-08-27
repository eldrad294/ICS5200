create table REP_EXECUTION_PLANS tablespace users as select * from v$sql where 1=0;
alter table REP_EXECUTION_PLANS add TPC_TRANSACTION_NAME varchar2(20);
alter table REP_EXECUTION_PLANS add STATEMENT_HASH_SUM varchar2(4000);
alter table REP_EXECUTION_PLANS add BENCHMARK_ITERATION varchar2(2);
