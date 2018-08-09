/*
--
-- ******************************
-- * Deleting Schema Statistics *
-- ******************************
--
* https://docs.oracle.com/cd/A91202_01/901_doc/appdev.901/a89852/dbms_16d.htm
* https://gist.github.com/davidhooey/6923444
*/
execute dbms_stats.delete_schema_stats(ownname=>'tpcds1');
execute dbms_stats.delete_schema_stats(ownname=>'tpcds10');
execute dbms_stats.delete_schema_stats(ownname=>'tpcds50');
execute dbms_stats.delete_schema_stats(ownname=>'tpcds100');