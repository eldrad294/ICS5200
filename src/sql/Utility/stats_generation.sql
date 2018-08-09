/* 
--
-- ********************************
-- * Generating Schema Statistics *
-- ********************************
--
* https://docs.oracle.com/cd/A91202_01/901_doc/appdev.901/a89852/dbms_s34.htm
* http://www.dba-oracle.com/oracle_tips_dbms_stats1.htm
* https://gist.github.com/davidhooey/6923444
*/
execute dbms_stats.gather_schema_stats(ownname=>'tpcds1',
									   estimate_percent=>DBMS_STATS.AUTO_SAMPLE_SIZE,
									   granularity=>'ALL',
									   cascade=>TRUE,
									   degree=>60,
									   method_opt=>'FOR ALL COLUMNS',
									   options=>'GATHER');
execute dbms_stats.gather_schema_stats(ownname=>'tpcds10',
									   estimate_percent=>DBMS_STATS.AUTO_SAMPLE_SIZE,
									   granularity=>'ALL',
									   cascade=>TRUE,
									   degree=>60,
									   method_opt=>'FOR ALL COLUMNS',
									   options=>'GATHER');
execute dbms_stats.gather_schema_stats(ownname=>'tpcds50',
									   estimate_percent=>DBMS_STATS.AUTO_SAMPLE_SIZE,
									   granularity=>'ALL',
									   cascade=>TRUE,
									   degree=>60,
									   method_opt=>'FOR ALL COLUMNS',
									   options=>'GATHER');
execute dbms_stats.gather_schema_stats(ownname=>'tpcds100',
									   estimate_percent=>DBMS_STATS.AUTO_SAMPLE_SIZE,
									   granularity=>'ALL',
									   cascade=>TRUE,
									   degree=>60,
									   method_opt=>'FOR ALL COLUMNS',
									   options=>'GATHER'); 