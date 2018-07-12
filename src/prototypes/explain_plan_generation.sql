/*
Documentation
* https://dba.stackexchange.com/questions/84012/what-are-the-omem-and-1mem-columns-in-the-plan-output
* http://www.dba-oracle.com/t_gather_plan_statistics.htm
*/

--explain plan for
select /*+ gather_plan_statistics */ *
from E_ACCOUNT_PERMISSION
where ap_ca_id = 0001
order by ap_acl desc;
--
SELECT *
FROM table(DBMS_XPLAN.DISPLAY_CURSOR(FORMAT=>'ALLSTATS LAST'));
--
SELECT *
FROM table(DBMS_XPLAN.DISPLAY_CURSOR(FORMAT=>'ALL'));
