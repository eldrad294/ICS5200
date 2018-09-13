LOAD DATA
INFILE '/home/gabriels/ICS5200/src/sql/Runtime/TPC-DS/tpcds1/Benchmark/rep_execution_plans.csv'
REPLACE
INTO TABLE tpcds1.rep_execution_plans
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
SQL_TEXT,
SQL_FULLTEXT,
SQL_ID,
SHARABLE_MEM integer external,
PERSISTENT_MEM integer external,
RUNTIME_MEM integer external,
SORTS integer external,
LOADED_VERSIONS integer external,
OPEN_VERSIONS integer external,
USERS_OPENING integer external,
FETCHES integer external,
EXECUTIONS integer external,
PX_SERVERS_EXECUTIONS integer external,
END_OF_FETCH_COUNT integer external,
USERS_EXECUTING integer external,
LOADS integer external,
FIRST_LOAD_TIME,
INVALIDATIONS integer external,
PARSE_CALLS integer external,
DISK_READS integer external,
DIRECT_WRITES integer external,
BUFFER_GETS integer external,
APPLICATION_WAIT_TIME integer external,
CONCURRENCY_WAIT_TIME integer external,
CLUSTER_WAIT_TIME  integer external,
USER_IO_WAIT_TIME integer external,
PLSQL_EXEC_TIME, integer external,
JAVA_EXEC_TIME integer external,
ROWS_PROCESSED integer external,
COMMAND_TYPE integer external,
OPTIMIZER_MODE,
OPTIMIZER_COST integer external,
OPTIMIZER_ENV,
OPTIMIZER_ENV_HASH_VALUE integer external,
PARSING_USER_ID integer external,
PARSING_SCHEMA_ID, integer external,
PARSING_SCHEMA_NAME,
KEPT_VERSIONS integer external,
ADDRESS,
TYPE_CHK_HEAP,
HASH_VALUE integer external,
OLD_HASH_VALUE integer external,
PLAN_HASH_VALUE integer external,
FULL_PLAN_HASH_VALUE integer external,
CHILD_NUMBER integer external,
SERVICE,
SERVICE_HASH integer external,
MODULE,
MODULE_HASH integer external,
ACTION,
ACTION_HASH integer external,
SERIALIZABLE_ABORTS integer external,
OUTLINE_CATEGORY,
CPU_TIME integer external,
ELAPSED_TIME integer external,
OUTLINE_SID, integer external,
CHILD_ADDRESS,
SQLTYPE integer external,
REMOTE,
OBJECT_STATUS,
LITERAL_HASH_VALUE integer external,
LAST_LOAD_TIME,
IS_OBSOLETE,
IS_BIND_SENSITIVE,
IS_BIND_AWARE,
IS_SHAREABLE,
CHILD_LATCH integer external,
SQL_PROFILE,
SQL_PATCH,
SQL_PLAN_BASELINE,
PROGRAM_ID integer external,
PROGRAM_LINE# integer external,
EXACT_MATCHING_SIGNATURE integer external,
FORCE_MATCHING_SIGNATURE integer external,
LAST_ACTIVE_TIME,
BIND_DATA,
TYPECHECK_MEM integer external,
IO_CELL_OFFLOAD_ELIGIBLE_BYTES integer external,
IO_INTERCONNECT_BYTES integer external,
PHYSICAL_READ_REQUESTS integer external,
PHYSICAL_READ_BYTES integer external,
PHYSICAL_WRITE_REQUESTS, integer external,
PHYSICAL_WRITE_BYTES integer external,
OPTIMIZED_PHY_READ_REQUESTS integer external,
LOCKED_TOTAL, integer external,
PINNED_TOTAL integer external,
IO_CELL_UNCOMPRESSED_BYTES integer external,
IO_CELL_OFFLOAD_RETURNED_BYTES integer external,
CON_ID, integer external,
IS_REOPTIMIZABLE,
IS_RESOLVED_ADAPTIVE_PLAN,
IM_SCANS integer external,
IM_SCAN_BYTES_UNCOMPRESSED integer external,
IM_SCAN_BYTES_INMEMORY integer external,
TPC_TRANSACTION_NAME,
STATEMENT_HASH_SUM,
BENCHMARK_ITERATION,
GATHERED_STATS
)