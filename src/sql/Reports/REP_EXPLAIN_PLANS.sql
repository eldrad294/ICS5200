create table REP_EXPLAIN_PLANS tablespace users as select * from plan_table where 1=0;
alter table REP_EXPLAIN_PLANS add TPC_TRANSACTION_NAME varchar2(20);
alter table REP_EXPLAIN_PLANS add STATEMENT_HASH_SUM varchar2(4000);
alter table REP_EXPLAIN_PLANS add BENCHMARK_ITERATION varchar2(2);
alter table REP_EXPLAIN_PLANS add GATHERED_STATS varchar2(5);
--
drop table REP_EXECUTION_PLANS;