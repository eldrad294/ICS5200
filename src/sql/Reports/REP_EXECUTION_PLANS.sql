create table REP_EXECUTION_PLANS tablespace users as select * from v$sql where 1=0;
alter table REP_EXECUTION_PLANS add TPC_TRANSACTION_NAME varchar2(20);
alter table REP_EXECUTION_PLANS add STATEMENT_HASH_SUM varchar2(4000);
alter table REP_EXECUTION_PLANS add BENCHMARK_ITERATION varchar2(2);
alter table REP_EXECUTION_PLANS add GATHERED_STATS varchar2(5);
--
drop table REP_EXECUTION_PLANS;
--
insert into REP_EXECUTION_PLANS
select *
from(
  select vs.*,
  		'place_holder_transaction_name',
  		'place_holder_md5_sum',
  		'place_holder_iteration_run',
  		'place_holder_gathered_stats'
  from v$sql vs
  where sql_text like '%" + self.__execution_plan_hint + "%'
  and sql_text not like '%v_sql%'
  and sql_text not like '%V_SQL%'
  order by last_active_time desc
) where rownum <= 1;

select tpc_transaction_name as tpc_transaction_name,  count(tpc_transaction_name) as number_of_executions,  sum(SHARABLE_MEM) / 5 as SHARABLE_MEM,  sum(PERSISTENT_MEM) / 5 as PERSISTENT_MEM,  sum(RUNTIME_MEM) / 5 as RUNTIME_MEM,  sum(SORTS) / 5 as SORTS,  sum(LOADED_VERSIONS) / 5 as LOADED_VERSIONS,  sum(OPEN_VERSIONS) / 5 as OPEN_VERSIONS,  sum(USERS_OPENING) / 5 as USERS_OPENING,  sum(FETCHES) / 5 as FETCHES,  sum(EXECUTIONS) / 5 as EXECUTIONS,  sum(PX_SERVERS_EXECUTIONS) / 5 as PX_SERVERS_EXECUTIONS,  sum(END_OF_FETCH_COUNT) / 5 as END_OF_FETCH_COUNT,  sum(USERS_EXECUTING) / 5 as USERS_EXECUTING,  sum(LOADS) / 5 as LOADS,  min(FIRST_LOAD_TIME) as FIRST_LOAD_TIME,  sum(INVALIDATIONS) / 5 as INVALIDATIONS,  sum(PARSE_CALLS) / 5 as PARSE_CALLS,  sum(DISK_READS) / 5 as DISK_READS,  sum(DIRECT_WRITES) / 5 as DIRECT_WRITES,  sum(BUFFER_GETS) / 5 as BUFFER_GETS,  sum(APPLICATION_WAIT_TIME) / 5 as APPLICATION_WAIT_TIME,  sum(CONCURRENCY_WAIT_TIME) / 5 as CONCURRENCY_WAIT_TIME,  sum(CLUSTER_WAIT_TIME) / 5 as CLUSTER_WAIT_TIME,  sum(USER_IO_WAIT_TIME) / 5 as USER_IO_WAIT_TIME,  sum(PLSQL_EXEC_TIME) / 5 as PLSQL_EXEC_TIME,  sum(JAVA_EXEC_TIME) / 5 as JAVA_EXEC_TIME,  sum(OPTIMIZER_COST) / 5 as OPTIMIZER_COST,  sum(CHILD_NUMBER) / 5 as CHILD_NUMBER,  sum(SERIALIZABLE_ABORTS) / 5 as SERIALIZABLE_ABORTS,  sum(OUTLINE_CATEGORY) / 5 as OUTLINE_CATEGORY,  sum(CPU_TIME) / 5 as CPU_TIME,  round((sum(ELAPSED_TIME) / 5) / (1000*60*60)) as ELAPSED_TIME_MINS,  sum(OUTLINE_SID) / 5 as OUTLINE_SID,  sum(SQLTYPE) / 5 as SQLTYPE,  min(LAST_LOAD_TIME) as LAST_LOAD_TIME,  sum(CHILD_LATCH) / 5 as CHILD_LATCH,  min(LAST_ACTIVE_TIME) as LAST_ACTIVE_TIME,  sum(TYPECHECK_MEM) / 5 as TYPECHECK_MEM,  sum(IO_CELL_OFFLOAD_ELIGIBLE_BYTES) / 5 as IO_CELL_OFFLOAD_ELIGIBLE_BYTES,  sum(IO_INTERCONNECT_BYTES) / 5 as IO_INTERCONNECT_BYTES,  sum(PHYSICAL_READ_REQUESTS) / 5 as PHYSICAL_READ_REQUESTS,  sum(PHYSICAL_READ_BYTES) / 5 as PHYSICAL_READ_BYTES,  sum(PHYSICAL_WRITE_REQUESTS) / 5 as PHYSICAL_WRITE_REQUESTS,  sum(PHYSICAL_WRITE_BYTES) / 5 as PHYSICAL_WRITE_BYTES,  sum(OPTIMIZED_PHY_READ_REQUESTS) / 5 as OPTIMIZED_PHY_READ_REQUESTS,  sum(LOCKED_TOTAL) / 5 as LOCKED_TOTAL,  sum(PINNED_TOTAL) / 5 as PINNED_TOTAL,  sum(IO_CELL_UNCOMPRESSED_BYTES) / 5 as IO_CELL_UNCOMPRESSED_BYTES,  sum(IO_CELL_OFFLOAD_RETURNED_BYTES) / 5 as IO_CELL_OFFLOAD_RETURNED_BYTES,  sum(IM_SCANS) / 5 as IM_SCANS,  sum(IM_SCAN_BYTES_UNCOMPRESSED) / 5 as IM_SCAN_BYTES_UNCOMPRESSED,  sum(IM_SCAN_BYTES_INMEMORY) / 5 as IM_SCAN_BYTES_INMEMORY,  count(STATEMENT_HASH_SUM) as STATEMENT_HASH_SUM,  count(BENCHMARK_ITERATION) as BENCHMARK_ITERATIONS  from REP_EXECUTION_PLANS  where GATHERED_STATS = 'False'  group by tpc_transaction_name  order by first_load_time,  tpc_transaction_name;