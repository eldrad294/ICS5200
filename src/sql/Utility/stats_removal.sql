/*
--
-- ******************************
-- * Deleting Schema Statistics *
-- ******************************
--
* https://docs.oracle.com/cd/A91202_01/901_doc/appdev.901/a89852/dbms_16d.htm
* https://gist.github.com/davidhooey/6923444
*/
execute dbms_stats.delete_database_stats(statown=>'tpcds1');
execute dbms_stats.delete_database_stats(statown=>'tpcds10');
--execute dbms_stats.delete_database_stats(statown=>'tpcds50');
execute dbms_stats.delete_database_stats(statown=>'tpcds100');
