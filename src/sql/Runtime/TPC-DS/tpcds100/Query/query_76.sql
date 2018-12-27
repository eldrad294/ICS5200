select * from (select  channel, col_name, d_year, d_qoy, i_category, COUNT(*) sales_cnt, SUM(ext_sales_price) sales_amt FROM (
        SELECT 'store' as channel, 'ss_customer_sk' col_name, d_year, d_qoy, i_category, ss_ext_sales_price ext_sales_price
         FROM store_sales, item, date_dim
         WHERE ss_customer_sk IS NULL
           AND ss_sold_date_sk=d_date_sk
           AND ss_item_sk=i_item_sk
           and i_item_sk between 584 and 600
           and d_date_sk between 2415522 and 2425522
           and rownum <= 50
        UNION ALL
        SELECT 'web' as channel, 'ws_web_site_sk' col_name, d_year, d_qoy, i_category, ws_ext_sales_price ext_sales_price
         FROM web_sales, item, date_dim
         WHERE ws_web_site_sk IS NULL
           AND ws_sold_date_sk=d_date_sk
           AND ws_item_sk=i_item_sk
           and d_date_sk between 2415522 and 2425522
           and i_item_sk between 584 and 600
           and rownum <= 50
        UNION ALL
        SELECT 'catalog' as channel, 'cs_bill_addr_sk' col_name, d_year, d_qoy, i_category, cs_ext_sales_price ext_sales_price
         FROM catalog_sales, item, date_dim
         WHERE cs_bill_addr_sk IS NULL
           AND cs_sold_date_sk=d_date_sk
           AND cs_item_sk=i_item_sk
           and i_item_sk between 584 and 600
           and d_date_sk between 2415522 and 2425522
           and rownum <= 50) foo
GROUP BY channel, col_name, d_year, d_qoy, i_category
ORDER BY channel, col_name, d_year, d_qoy, i_category
 ) where rownum <= 100;