WITH ss
     AS (SELECT i_item_id,
                Sum(ss_ext_sales_price) total_sales
         FROM   store_sales,
                date_dim,
                customer_address,
                item
         WHERE  i_item_id IN (SELECT i_item_id
                              FROM   item
                              WHERE  i_color IN ( 'khaki', 'powder', 'red' ))
                AND ss_item_sk = i_item_sk
                AND ss_sold_date_sk = d_date_sk
                AND d_year = 2002
                AND d_moy = 5
                AND ss_addr_sk = ca_address_sk
                AND ca_gmt_offset = -8
                and rownum <= 10000
         GROUP  BY i_item_id),
     cs
     AS (SELECT i_item_id,
                Sum(cs_ext_sales_price) total_sales
         FROM   catalog_sales,
                date_dim,
                customer_address,
                item
         WHERE  i_item_id IN (SELECT i_item_id
                              FROM   item
                              WHERE  i_color IN ( 'khaki', 'powder', 'red' ))
                AND cs_item_sk = i_item_sk
                AND cs_sold_date_sk = d_date_sk
                AND d_year = 2002
                AND d_moy = 5
                AND cs_bill_addr_sk = ca_address_sk
                AND ca_gmt_offset = -8
                and rownum <= 10000
         GROUP  BY i_item_id),
     ws
     AS (SELECT i_item_id,
                Sum(ws_ext_sales_price) total_sales
         FROM   web_sales,
                date_dim,
                customer_address,
                item
         WHERE  i_item_id IN (SELECT i_item_id
                              FROM   item
                              WHERE  i_color IN ( 'khaki', 'powder', 'red' ))
                AND ws_item_sk = i_item_sk
                AND ws_sold_date_sk = d_date_sk
                AND d_year = 2002
                AND d_moy = 5
                AND ws_bill_addr_sk = ca_address_sk
                AND ca_gmt_offset = -8
                and rownum <= 10000
         GROUP  BY i_item_id)
SELECT *
FROM   (SELECT i_item_id,
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
        GROUP  BY i_item_id
        ORDER  BY total_sales,
                  i_item_id)
WHERE  rownum <= 100;