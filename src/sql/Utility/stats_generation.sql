/*
* https://docs.oracle.com/cd/A91202_01/901_doc/appdev.901/a89852/dbms_s34.htm
* http://www.dba-oracle.com/oracle_tips_dbms_stats1.htm
* https://gist.github.com/davidhooey/6923444
*/
--
-- ********************************
-- * Generating Schema Statistics *
-- ********************************
--
execute dbms_stats.gather_database_stats(statown=>'tpcds1',
									     estimate_percent=>DBMS_STATS.AUTO_SAMPLE_SIZE,
									     granularity=>'ALL',
									     cascade=>TRUE,
									     degree=>60,
									     method_opt=>'FOR ALL COLUMNS',
									     options=>'GATHER',
									     gather_sys=>TRUE,
									     no_invalidate=>DBMS_STATS.AUTO_INVALIDATE);
