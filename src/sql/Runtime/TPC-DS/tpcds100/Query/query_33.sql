WITH ss 
     AS (SELECT i_manufact_id, 
                Sum(ss_ext_sales_price) total_sales 
         FROM   store_sales, 
                date_dim, 
                customer_address, 
                item 
         WHERE  i_manufact_id IN (SELECT i_manufact_id 
                                  FROM   item 
                                  WHERE  i_category IN ( 'Home' )) 
                AND ss_item_sk = i_item_sk 
                AND ss_sold_date_sk = d_date_sk 
                AND d_year = 2002 
                AND d_moy = 1 
                AND ss_addr_sk = ca_address_sk 
                AND ca_gmt_offset = -5
                and rownum <= 10000
         GROUP  BY i_manufact_id), 
     cs 
     AS (SELECT i_manufact_id, 
                Sum(cs_ext_sales_price) total_sales 
         FROM   catalog_sales, 
                date_dim, 
                customer_address, 
                item 
         WHERE  i_manufact_id IN (SELECT i_manufact_id 
                                  FROM   item 
                                  WHERE  i_category IN ( 'Home' )) 
                AND cs_item_sk = i_item_sk 
                AND cs_sold_date_sk = d_date_sk 
                AND d_year = 2002 
                AND d_moy = 1 
                AND cs_bill_addr_sk = ca_address_sk 
                AND ca_gmt_offset = -5
                and rownum <= 10000
         GROUP  BY i_manufact_id), 
     ws 
     AS (SELECT i_manufact_id, 
                Sum(ws_ext_sales_price) total_sales 
         FROM   web_sales, 
                date_dim, 
                customer_address, 
                item 
         WHERE  i_manufact_id IN (SELECT i_manufact_id 
                                  FROM   item 
                                  WHERE  i_category IN ( 'Home' )) 
                AND ws_item_sk = i_item_sk 
                AND ws_sold_date_sk = d_date_sk 
                AND d_year = 2002 
                AND d_moy = 1 
                AND ws_bill_addr_sk = ca_address_sk 
                AND ca_gmt_offset = -5
                and rownum <= 10000
         GROUP  BY i_manufact_id) 
SELECT * 
FROM   (SELECT i_manufact_id, 
               Sum(total_sales) total_sales 
        FROM   (SELECT * 
                FROM   ss
                where rownum <= 10000
                UNION ALL 
                SELECT * 
                FROM   cs
                where rownum <= 10000
                UNION ALL 
                SELECT * 
                FROM   ws
                where rownum <= 10000) tmp1
        where rownum <= 10000
        GROUP  BY i_manufact_id 
        ORDER  BY total_sales) 
WHERE  rownum <= 100; 