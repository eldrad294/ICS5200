select * from (select  
   count(distinct ws_order_number) as "order count"
  ,sum(ws_ext_ship_cost) as "total shipping cost"
  ,sum(ws_net_profit) as "total net profit"
from
   web_sales ws1
  ,date_dim
  ,customer_address
  ,web_site
where
    d_date between to_char(to_date('2002-4-01','yyyy/mm/dd'),'yyyy-mm-dd') and
           (to_char(to_date('2002-4-01','yyyy/mm/dd') + 60,'yyyy-mm-dd'))
and ws1.ws_ship_date_sk = d_date_sk
and ws1.ws_ship_addr_sk = ca_address_sk
and ca_state = 'NY'
and ws1.ws_web_site_sk = web_site_sk
and web_company_name = 'pri'
and rownum <= 10000
and exists (select *
            from web_sales ws2
            where ws1.ws_order_number = ws2.ws_order_number
              and ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk
              and rownum <= 1)
and not exists(select *
               from web_returns wr1
               where ws1.ws_order_number = wr1.wr_order_number
               and rownum <= 1)
order by count(distinct ws_order_number)
 ) where rownum <= 100;