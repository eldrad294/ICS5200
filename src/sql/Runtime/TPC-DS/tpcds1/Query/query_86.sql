    select * from (select  i_item_id        ,i_item_desc        ,i_current_price  from item, inventory, date_dim, store_sales  where i_current_price between 34 and 34+30  and inv_item_sk = i_item_sk  and d_date_sk=inv_date_sk  and d_date between to_date2002-01-17' ,'yyyy-mm-dd') and (to_date2002-01-17' ,'yyyy-mm-dd') +  60)  and i_manufact_id in (530,987,259,487)  and inv_quantity_on_hand between 100 and 500  and ss_item_sk = i_item_sk  group by i_item_id,i_item_desc,i_current_price  order by i_item_id   ) where rownum <= 100