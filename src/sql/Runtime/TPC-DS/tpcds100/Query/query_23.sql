WITH frequent_ss_items 
     AS (SELECT Substr(i_item_desc, 1, 30) itemdesc, 
                i_item_sk                  item_sk, 
                d_date                     solddate, 
                Count(*)                   cnt 
         FROM   store_sales, 
                date_dim, 
                item 
         WHERE  ss_sold_date_sk = d_date_sk 
                AND ss_item_sk = i_item_sk 
                AND d_year IN ( 2000, 2000 + 1, 2000 + 2, 2000 + 3 )
                and rownum <= 10000
         GROUP  BY Substr(i_item_desc, 1, 30), 
                   i_item_sk, 
                   d_date 
         HAVING Count(*) > 4), 
     max_store_sales 
     AS (SELECT Max(csales) tpcds_cmax 
         FROM   (SELECT c_customer_sk, 
                        Sum(ss_quantity * ss_sales_price) csales 
                 FROM   store_sales, 
                        customer, 
                        date_dim 
                 WHERE  ss_customer_sk = c_customer_sk 
                        AND ss_sold_date_sk = d_date_sk 
                        AND d_year IN ( 2000, 2000 + 1, 2000 + 2, 2000 + 3 )
                        and rownum <= 10000
                 GROUP  BY c_customer_sk)), 
     best_ss_customer 
     AS (SELECT c_customer_sk, 
                Sum(ss_quantity * ss_sales_price) ssales 
         FROM   store_sales, 
                customer 
         WHERE  ss_customer_sk = c_customer_sk
         and rownum <= 10000
         GROUP  BY c_customer_sk 
         HAVING Sum(ss_quantity * ss_sales_price) > 
                ( 95 / 100.0 ) * (SELECT * 
                                  FROM   max_store_sales
                                  where rownum <= 10000 ))
SELECT * 
FROM   (SELECT Sum(sales) 
        FROM   (SELECT cs_quantity * cs_list_price sales 
                FROM   catalog_sales, 
                       date_dim 
                WHERE  d_year = 2000 
                       AND d_moy = 5 
                       AND cs_sold_date_sk = d_date_sk 
                       AND cs_item_sk IN (SELECT item_sk 
                                          FROM   frequent_ss_items
                                          where rownum <= 10000 )
                       AND cs_bill_customer_sk IN (SELECT c_customer_sk 
                                                   FROM   best_ss_customer
                                                   where rownum <= 10000 )
                       and rownum <= 10000
                UNION ALL 
                SELECT ws_quantity * ws_list_price sales 
                FROM   web_sales, 
                       date_dim 
                WHERE  d_year = 2000 
                       AND d_moy = 5 
                       AND ws_sold_date_sk = d_date_sk 
                       AND ws_item_sk IN (SELECT item_sk 
                                          FROM   frequent_ss_items
                                          where rownum <= 10000 )
                       AND ws_bill_customer_sk IN (SELECT c_customer_sk 
                                                   FROM   best_ss_customer
                                                   where rownum <= 10000 )))
                       and rownum <= 10000
WHERE  rownum <= 100; 

WITH frequent_ss_items 
     AS (SELECT Substr(i_item_desc, 1, 30) itemdesc, 
                i_item_sk                  item_sk, 
                d_date                     solddate, 
                Count(*)                   cnt 
         FROM   store_sales, 
                date_dim, 
                item 
         WHERE  ss_sold_date_sk = d_date_sk 
                AND ss_item_sk = i_item_sk 
                AND d_year IN ( 2000, 2000 + 1, 2000 + 2, 2000 + 3 )
                and rownum <= 10000
         GROUP  BY Substr(i_item_desc, 1, 30), 
                   i_item_sk, 
                   d_date 
         HAVING Count(*) > 4), 
     max_store_sales 
     AS (SELECT Max(csales) tpcds_cmax 
         FROM   (SELECT c_customer_sk, 
                        Sum(ss_quantity * ss_sales_price) csales 
                 FROM   store_sales, 
                        customer, 
                        date_dim 
                 WHERE  ss_customer_sk = c_customer_sk 
                        AND ss_sold_date_sk = d_date_sk 
                        AND d_year IN ( 2000, 2000 + 1, 2000 + 2, 2000 + 3 )
                        and rownum <= 10000
                 GROUP  BY c_customer_sk)), 
     best_ss_customer 
     AS (SELECT c_customer_sk, 
                Sum(ss_quantity * ss_sales_price) ssales 
         FROM   store_sales, 
                customer 
         WHERE  ss_customer_sk = c_customer_sk
         and rownum <= 10000
         GROUP  BY c_customer_sk 
         HAVING Sum(ss_quantity * ss_sales_price) > 
                ( 95 / 100.0 ) * (SELECT * 
                                  FROM   max_store_sales)) 
SELECT * 
FROM   (SELECT c_last_name, 
               c_first_name, 
               sales 
        FROM   (SELECT c_last_name, 
                       c_first_name, 
                       Sum(cs_quantity * cs_list_price) sales 
                FROM   catalog_sales, 
                       customer, 
                       date_dim 
                WHERE  d_year = 2000 
                       AND d_moy = 5 
                       AND cs_sold_date_sk = d_date_sk 
                       AND cs_item_sk IN (SELECT item_sk 
                                          FROM   frequent_ss_items) 
                       AND cs_bill_customer_sk IN (SELECT c_customer_sk 
                                                   FROM   best_ss_customer) 
                       AND cs_bill_customer_sk = c_customer_sk
                       and rownum <= 10000
                GROUP  BY c_last_name, 
                          c_first_name 
                UNION ALL 
                SELECT c_last_name, 
                       c_first_name, 
                       Sum(ws_quantity * ws_list_price) sales 
                FROM   web_sales, 
                       customer, 
                       date_dim 
                WHERE  d_year = 2000 
                       AND d_moy = 5 
                       AND ws_sold_date_sk = d_date_sk 
                       AND ws_item_sk IN (SELECT item_sk 
                                          FROM   frequent_ss_items) 
                       AND ws_bill_customer_sk IN (SELECT c_customer_sk 
                                                   FROM   best_ss_customer) 
                       AND ws_bill_customer_sk = c_customer_sk
                       and rownum <= 10000
                GROUP  BY c_last_name, 
                          c_first_name) 
        ORDER  BY c_last_name, 
                  c_first_name, 
                  sales) 
WHERE  rownum <= 100; 