select * from (select
  ca_state,
  cd_gender,
  cd_marital_status,
  cd_dep_count,
  count(*) cnt1,
  max(cd_dep_count) as cd1,
  sum(cd_dep_count) as cd2,
  max(cd_dep_count) as cd3,
  cd_dep_employed_count,
  count(*) cnt2,
  max(cd_dep_employed_count) as cd4,
  sum(cd_dep_employed_count) as cd5,
  max(cd_dep_employed_count) as cd6,
  cd_dep_college_count,
  count(*) cnt3,
  max(cd_dep_college_count) as cd7,
  sum(cd_dep_college_count) as cd8,
  max(cd_dep_college_count) as cd9
 from
  customer c,customer_address ca,customer_demographics
 where
  c.c_current_addr_sk = ca.ca_address_sk and
  c_customer_sk between 579 and 900 and
  cd_demo_sk = c.c_current_cdemo_sk and
  rownum <= 10000 and
  exists (select *
          from store_sales,date_dim
          where c.c_customer_sk = ss_customer_sk and
                ss_sold_date_sk = d_date_sk and
                d_year = 2001 and
                d_qoy < 4
                and rownum <= 1) and
   (exists (select *
            from web_sales,date_dim
            where c.c_customer_sk = ws_bill_customer_sk and
                  ws_sold_date_sk = d_date_sk and
                  d_year = 2001 and
                  d_qoy < 4
                  and rownum <= 1) or
    exists (select *
            from catalog_sales,date_dim
            where c.c_customer_sk = cs_ship_customer_sk and
                  cs_sold_date_sk = d_date_sk and
                  d_year = 2001 and
                  d_qoy < 4
                  and rownum <= 1))
 group by ca_state,
          cd_gender,
          cd_marital_status,
          cd_dep_count,
          cd_dep_employed_count,
          cd_dep_college_count
 order by ca_state,
          cd_gender,
          cd_marital_status,
          cd_dep_count,
          cd_dep_employed_count,
          cd_dep_college_count
  ) where rownum <= 100;