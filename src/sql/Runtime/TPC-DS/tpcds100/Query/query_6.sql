SELECT *
FROM   (SELECT a.ca_state state,
               Count(*)   cnt
        FROM   customer_address a,
               customer c,
               store_sales s,
               date_dim d,
               item i
        WHERE  a.ca_address_sk = c.c_current_addr_sk
               AND c.c_customer_sk = s.ss_customer_sk
               AND s.ss_sold_date_sk = d.d_date_sk
               AND s.ss_item_sk = i.i_item_sk
               AND d.d_month_seq = (SELECT DISTINCT ( d_month_seq )
                                    FROM   date_dim
                                    WHERE  d_year = 2002
                                    and d_date_sk between 2415522 and 2415523
                                           AND d_moy = 3)
               AND i.i_current_price > 1.2 *
                   (SELECT Avg(j.i_current_price)
                    FROM   item j
                    where j.i_item_sk between 581 and 690
                    and  j.i_category = i.i_category)
               and a.ca_address_sk between 582 and 999
               and i.i_item_sk between 581 and 690
               and d.d_date_sk between 2415522 and 2415523
               and rownum <= 10000
        GROUP  BY a.ca_state
        HAVING Count(*) >= 10
        ORDER  BY cnt,
                  a.ca_state)
WHERE  rownum <= 100;
