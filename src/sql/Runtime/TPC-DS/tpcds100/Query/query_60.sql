WITH ss
     AS (SELECT i_item_id,
                Sum(ss_ext_sales_price) total_sales
         FROM   store_sales,
                date_dim,
                customer_address,
                item
         WHERE  i_item_id IN (SELECT i_item_id
                              FROM   item
                              WHERE  i_category IN ( 'Jewelry' )
                              and i_item_sk between 579 and 600
                              and rownum <= 100)
                AND ss_item_sk = i_item_sk
                and i_item_sk between 579 and 600
                AND ss_sold_date_sk = d_date_sk
                AND d_year = 2001
                AND d_moy = 8
                AND ss_addr_sk = ca_address_sk
                AND ca_gmt_offset = -6
                and rownum <= 100
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
                              WHERE  i_category IN ( 'Jewelry' )
                              and i_item_sk between 579 and 600
                              and rownum <= 100)
                AND cs_item_sk = i_item_sk
                and i_item_sk between 579 and 600
                AND cs_sold_date_sk = d_date_sk
                AND d_year = 2001
                AND d_moy = 8
                AND cs_bill_addr_sk = ca_address_sk
                AND ca_gmt_offset = -6
                and rownum <= 100
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
                              WHERE  i_category IN ( 'Jewelry' )
                              and i_item_sk between 579 and 600
                              and rownum <= 100)
                AND ws_item_sk = i_item_sk
                and i_item_sk between 579 and 600
                AND ws_sold_date_sk = d_date_sk
                AND d_year = 2001
                AND d_moy = 8
                AND ws_bill_addr_sk = ca_address_sk
                AND ca_gmt_offset = -6
                and rownum <= 100
         GROUP  BY i_item_id)
SELECT *
FROM   (SELECT i_item_id,
               Sum(total_sales) total_sales
        FROM   (SELECT *
                FROM   ss
                where rownum <= 100
                UNION ALL
                SELECT *
                FROM   cs
                where rownum <= 100
                UNION ALL
                SELECT *
                FROM   ws
                where rownum <= 100) tmp1
        where rownum <= 100
        GROUP  BY i_item_id
        ORDER  BY i_item_id,
                  total_sales)
WHERE  rownum <= 100;