select * from (select     substr(w_warehouse_name,1,20)  ,sm_type  ,cc_name  ,sum(case when (cs_ship_date_sk - cs_sold_date_sk <= 30 ) then 1 else 0 end)  as "30 days"   ,sum(case when (cs_ship_date_sk - cs_sold_date_sk > 30) and                  (cs_ship_date_sk - cs_sold_date_sk <= 60) then 1 else 0 end )  as "31-60 days"   ,sum(case when (cs_ship_date_sk - cs_sold_date_sk > 60) and                  (cs_ship_date_sk - cs_sold_date_sk <= 90) then 1 else 0 end)  as "61-90 days"   ,sum(case when (cs_ship_date_sk - cs_sold_date_sk > 90) and                 (cs_ship_date_sk - cs_sold_date_sk <= 120) then 1 else 0 end)  as "91-120 days"   ,sum(case when (cs_ship_date_sk - cs_sold_date_sk  > 120) then 1 else 0 end)  as ">120 days" from   catalog_sales  ,warehouse  ,ship_mode  ,call_center  ,date_dimwhere    d_month_seq between 1191 and 1191 + 11and cs_ship_date_sk   = d_date_skand cs_warehouse_sk   = w_warehouse_skand cs_ship_mode_sk   = sm_ship_mode_skand cs_call_center_sk = cc_call_center_skgroup by   substr(w_warehouse_name,1,20)  ,sm_type  ,cc_nameorder by substr(w_warehouse_name,1,20)        ,sm_type        ,cc_name ) where rownum <= 100;