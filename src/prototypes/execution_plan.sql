/*
Execution Plan
*/
select /*ICS5200_MONITOR_HINT*/ *
from CATALOG_SALES
where cs_sold_date_sk = '2450816'
order by cs_sold_time_sk;
--
select *
from v$sql_plan
where sql_id = (
  select sql_id
  from (
    select SQL_ID
    from v$sql
    where sql_fulltext like '%ICS5200_MONITOR_HINT%'
    and sql_fulltext not like '%v$sql%'
    and sql_fulltext not like '%V$SQL%'
    order by last_active_time desc
  ) where rownum = 1
) order by id;
