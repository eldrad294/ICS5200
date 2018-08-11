with ssci as (select ss_customer_sk customer_sk      ,ss_item_sk item_skfrom store_sales,date_dimwhere ss_sold_date_sk = d_date_sk  and d_month_seq between 1204 and 1204 + 11group by ss_customer_sk        ,ss_item_sk),csci as( select cs_bill_customer_sk customer_sk      ,cs_item_sk item_skfrom catalog_sales,date_dimwhere cs_sold_date_sk = d_date_sk  and d_month_seq between 1204 and 1204 + 11group by cs_bill_customer_sk        ,cs_item_sk)select * from ( select  sum(case when ssci.customer_sk is not null and csci.customer_sk is null then 1 else 0 end) store_only      ,sum(case when ssci.customer_sk is null and csci.customer_sk is not null then 1 else 0 end) catalog_only      ,sum(case when ssci.customer_sk is not null and csci.customer_sk is not null then 1 else 0 end) store_and_catalogfrom ssci full outer join csci on (ssci.customer_sk=csci.customer_sk                               and ssci.item_sk = csci.item_sk) ) where rownum <= 100;