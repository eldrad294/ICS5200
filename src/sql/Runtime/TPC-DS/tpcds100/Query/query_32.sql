select * from (select  sum(cs_ext_discount_amt)  as "excess discount amount" 
from 
   catalog_sales 
   ,item 
   ,date_dim
where
i_manufact_id = 29
and i_item_sk = cs_item_sk 
and d_date between to_char(to_date('1999-01-07','yyyy/mm/dd'),'yyyy-mm-dd') and
        (to_char(to_date('1999-01-07','yyyy/mm/dd') + 90,'yyyy-mm-dd'))
and d_date_sk = cs_sold_date_sk
and rownum <= 10000
and cs_ext_discount_amt  
     > ( 
         select 
            1.3 * avg(cs_ext_discount_amt) 
         from 
            catalog_sales 
           ,date_dim
         where 
              cs_item_sk = i_item_sk 
          and d_date between to_char(to_date('1999-01-07','yyyy/mm/dd'),'yyyy-mm-dd') and
                             (to_char(to_date('1999-01-07','yyyy/mm/dd') + 90,'yyyy-mm-dd'))
          and d_date_sk = cs_sold_date_sk
          and rownum <= 10000
      ) 
 ) where rownum <= 100;