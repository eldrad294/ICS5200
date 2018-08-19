drop table s_catalog_page;
create table s_catalog_page tablespace tpcds_benchmark as
(select CP_CATALOG_NUMBER cpag_catalog_number
       ,CP_CATALOG_PAGE_NUMBER cpag_catalog_page_number 
       ,CP_DEPARTMENT cpag_department
 from CATALOG_PAGE);
