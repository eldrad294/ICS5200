select * from (select  i_item_id
      ,i_item_desc 
      ,i_category 
      ,i_class 
      ,i_current_price
      ,sum(ws_ext_sales_price) as itemrevenue 
      ,sum(ws_ext_sales_price)*100/sum(sum(ws_ext_sales_price)) over
          (partition by i_class) as revenueratio
from	
	web_sales
    	,item 
    	,date_dim
where 
	ws_item_sk = i_item_sk 
  	and i_category in ('Men', 'Sports', 'Home')
  	and ws_sold_date_sk = d_date_sk
	and d_date between to_char(to_date('2001-03-04','yyyy/mm/dd'),'yyyy-mm-dd')
				and (to_char(to_date('2001-03-04','yyyy/mm/dd') + 30,'yyyy-mm-dd'))
group by 
	i_item_id
        ,i_item_desc 
        ,i_category
        ,i_class
        ,i_current_price
order by 
	i_category
        ,i_class
        ,i_item_id
        ,i_item_desc
        ,revenueratio
 ) where rownum <= 100;