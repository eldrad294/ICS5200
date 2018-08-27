create table REP_EXECUTION_PLANS tablespace users as select * from v$sql where 1=0;
alter table REP_EXECUTION_PLANS add TPC_TRANSACTION_NAME varchar2(20);
alter table REP_EXECUTION_PLANS add STATEMENT_HASH_SUM varchar2(4000);
alter table REP_EXECUTION_PLANS add BENCHMARK_ITERATION varchar2(2);
--
drop table REP_EXECUTION_PLANS;
--
insert into REP_EXECUTION_PLANS
select * from(
    select vs.*, 'query_1.sql', '4dd7b29ad1d1eddc7deecb11216895cf'
    from v$sql vs where sql_text like '%/*ICS5200_MONITOR_HINT*/%'
    and sql_text not like '%v_sql%'
    and sql_text not like '%V_SQL%'
    order by last_active_time desc )
where rownum <= 1
