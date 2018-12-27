select * from (select  dt.d_year
 	,item.i_brand_id brand_id
 	,item.i_brand brand
 	,sum(ss_ext_sales_price) ext_price
 from date_dim dt
     ,store_sales
     ,item
 where dt.d_date_sk = store_sales.ss_sold_date_sk
    and store_sales.ss_item_sk = item.i_item_sk
    and d_Date_sk between 2415522 and 2425522
    and item.i_manager_id = 1
    and dt.d_moy=11
    and dt.d_year=1998
    and rownum <= 10000
 group by dt.d_year
 	,item.i_brand
 	,item.i_brand_id
 order by dt.d_year
 	,ext_price desc
 	,brand_id
 ) where rownum <= 100 ;