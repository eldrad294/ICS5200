execute dbms_stats.gather_database_stats(statown=>'tpcds10',
								         estimate_percent=>DBMS_STATS.AUTO_SAMPLE_SIZE,
								  	     granularity=>'ALL',
									     cascade=>TRUE,
									     degree=>60,
									     method_opt=>'FOR ALL COLUMNS',
									     options=>'GATHER');