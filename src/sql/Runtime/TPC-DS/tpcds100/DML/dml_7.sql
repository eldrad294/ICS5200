drop table catv;
create table catv tablespace tpcds_benchmark as
(select CP_CATALOG_PAGE_ID cp_catalog_page_id
      ,startd.d_date_sk cp_start_date_sk
      ,endd.d_date_sk cp_end_date_sk
      ,CP_DEPARTMENT cp_department
      ,CP_CATALOG_NUMBER cp_catalog_number
      ,CP_DESCRIPTION cp_description
      ,CP_TYPE cp_type
from catalog_page
    ,date_dim startd
    ,date_dim endd
where to_char(CP_START_DATE_SK) = startd.d_date
  and to_char(CP_END_DATE_SK) = endd.d_date);
select count(*) from catv;
select count(*) from s_catalog_page;
