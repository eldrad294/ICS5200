/*
Documentation
* https://dba.stackexchange.com/questions/84012/what-are-the-omem-and-1mem-columns-in-the-plan-output
* http://www.dba-oracle.com/t_gather_plan_statistics.htm

Example 1 - Query Estimation
*/
explain plan for
select *
from CATALOG_SALES
where cs_sold_date_sk = '2450816'
order by cs_sold_time_sk;
--
select *
from plan_table
where plan_id = (
  select max(plan_id)
  from plan_table
  where to_date(to_char(timestamp,'MM/DD/YYYY'),'MM/DD/YYYY') = to_date(to_char(sysdate,'MM/DD/YYYY'),'MM/DD/YYYY')
)
order by id;
--
/*
Example 2 - Requires execution of the actual query
*/
select /*+ gather_plan_statistics */ *
from CATALOG_SALES
where cs_sold_date_sk = '2450816'
order by cs_sold_time_sk;
--
SET LINESIZE 130
SELECT *
FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(format => 'ALLSTATS LAST'));

