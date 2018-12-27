select case when (select count(*)
                  from store_sales
                  where ss_quantity between 1 and 20
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) > 2972190
            then (select avg(ss_ext_sales_price)
                  from store_sales
                  where ss_quantity between 1 and 20
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000)
            else (select avg(ss_net_profit)
                  from store_sales
                  where ss_quantity between 1 and 20
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) end bucket1 ,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 21 and 40
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) > 4505785
            then (select avg(ss_ext_sales_price)
                  from store_sales
                  where ss_quantity between 21 and 40
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000)
            else (select avg(ss_net_profit)
                  from store_sales
                  where ss_quantity between 21 and 40
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) end bucket2,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 41 and 60
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) > 1575726
            then (select avg(ss_ext_sales_price)
                  from store_sales
                  where ss_quantity between 41 and 60
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000)
            else (select avg(ss_net_profit)
                  from store_sales
                  where ss_quantity between 41 and 60
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) end bucket3,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 61 and 80
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) > 3188917
            then (select avg(ss_ext_sales_price)
                  from store_sales
                  where ss_quantity between 61 and 80
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000)
            else (select avg(ss_net_profit)
                  from store_sales
                  where ss_quantity between 61 and 80
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) end bucket4,
       case when (select count(*)
                  from store_sales
                  where ss_quantity between 81 and 100
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) > 3525216
            then (select avg(ss_ext_sales_price)
                  from store_sales
                  where ss_quantity between 81 and 100
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000)
            else (select avg(ss_net_profit)
                  from store_sales
                  where ss_quantity between 81 and 100
                  and ss_item_sk between 95700 and 98000
                  and ss_ticket_number between 36615 and 40000) end bucket5
from reason
where r_reason_sk = 1
and rownum <= 10000
;