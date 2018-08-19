select * from (select  
   sum(ws_ext_discount_amt)  as "Excess Discount Amount" 
from 
    web_sales 
   ,item 
   ,date_dim
where
i_manufact_id = 135
and i_item_sk = ws_item_sk 
and d_date between to_char(to_date('2001-01-27','yyyy/mm/dd'),'yyyy-mm-dd') and
        (to_char(to_date('2001-01-27','yyyy/mm/dd')+ 90,'yyyy-mm-dd'))
and d_date_sk = ws_sold_date_sk 
and ws_ext_discount_amt  
     > ( 
         SELECT 
            1.3 * avg(ws_ext_discount_amt) 
         FROM 
            web_sales 
           ,date_dim
         WHERE 
              ws_item_sk = i_item_sk 
          and d_date between to_char(to_date('2001-01-27','yyyy/mm/dd'),'yyyy-mm-dd') and
                             (to_char(to_date('2001-01-27','yyyy/mm/dd') + 90,'yyyy-mm-dd'))
          and d_date_sk = ws_sold_date_sk 
      ) 
order by sum(ws_ext_discount_amt)
 ) where rownum <= 100;