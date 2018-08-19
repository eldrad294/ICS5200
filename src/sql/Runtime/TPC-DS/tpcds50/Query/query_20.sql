select * from (select  i_item_id
       ,i_item_desc 
       ,i_category 
       ,i_class 
       ,i_current_price
       ,sum(cs_ext_sales_price) as itemrevenue 
       ,sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over
           (partition by i_class) as revenueratio
 from	catalog_sales
     ,item 
     ,date_dim
 where cs_item_sk = i_item_sk 
   and i_category in ('Sports', 'Children', 'Shoes')
   and cs_sold_date_sk = d_date_sk
 and d_date between to_char(to_date('1998-02-07','yyyy/mm/dd'),'yyyy-mm-dd')
 				and (to_char(to_date('1998-02-07','yyyy/mm/dd') + 30,'yyyy-mm-dd'))
 group by i_item_id
         ,i_item_desc 
         ,i_category
         ,i_class
         ,i_current_price
 order by i_category
         ,i_class
         ,i_item_id
         ,i_item_desc
         ,revenueratio
 ) where rownum <= 100;