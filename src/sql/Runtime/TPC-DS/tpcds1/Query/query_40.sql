select * from (select  
   w_state
  ,i_item_id
  ,sum(case when (to_char(to_date(d_date,'yyyy/mm/dd'),'yyyy-mm-dd') < to_char(to_date('2001-05-02','yyyy/mm/dd'),'yyyy-mm-dd'))
 		then cs_sales_price - coalesce(cr_refunded_cash,0) else 0 end) as sales_before
  ,sum(case when (to_char(to_date(d_date,'yyyy/mm/dd'),'yyyy-mm-dd') >= to_char(to_date('2001-05-02','yyyy/mm/dd'),'yyyy-mm-dd'))
 		then cs_sales_price - coalesce(cr_refunded_cash,0) else 0 end) as sales_after
 from
   catalog_sales left outer join catalog_returns on
       (cs_order_number = cr_order_number 
        and cs_item_sk = cr_item_sk)
  ,warehouse 
  ,item
  ,date_dim
 where
     i_current_price between 0.99 and 1.49
 and i_item_sk          = cs_item_sk
 and cs_warehouse_sk    = w_warehouse_sk 
 and cs_sold_date_sk    = d_date_sk
 and d_date between (to_char(to_date('2001-05-02','yyyy/mm/dd') - 30,'yyyy-mm-dd'))
                and (to_char(to_date('2001-05-02','yyyy/mm/dd') + 30,'yyyy-mm-dd'))
 group by
    w_state,i_item_id
 order by w_state,i_item_id
 ) where rownum <= 100;