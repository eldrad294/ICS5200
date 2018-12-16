WITH ssr 
     AS (SELECT s_store_id, 
                Sum(sales_price) AS sales, 
                Sum(profit)      AS profit, 
                Sum(return_amt)  AS returns, 
                Sum(net_loss)    AS profit_loss 
         FROM   (SELECT ss_store_sk              AS store_sk, 
                        ss_sold_date_sk          AS date_sk, 
                        ss_ext_sales_price       AS sales_price, 
                        ss_net_profit            AS profit, 
                        Cast(0 AS DECIMAL(7, 2)) AS return_amt, 
                        Cast(0 AS DECIMAL(7, 2)) AS net_loss 
                 FROM   store_sales 
                 WHERE  rownum <= 10000 
                 UNION ALL 
                 SELECT sr_store_sk              AS store_sk, 
                        sr_returned_date_sk      AS date_sk, 
                        Cast(0 AS DECIMAL(7, 2)) AS sales_price, 
                        Cast(0 AS DECIMAL(7, 2)) AS profit, 
                        sr_return_amt            AS return_amt, 
                        sr_net_loss              AS net_loss 
                 FROM   store_returns 
                 WHERE  rownum <= 10000) salesreturns, 
                date_dim, 
                store 
         WHERE  date_sk = d_date_sk 
                AND d_date BETWEEN To_char(To_date('2000-08-19', 'yyyy/mm/dd'), 
                                   'yyyy-mm-dd') 
                                   AND ( 
                                       To_char(To_date('2000-08-19', 
                                               'yyyy/mm/dd') + 
                                               14, 
                                       'yyyy-mm-dd') ) 
                AND store_sk = s_store_sk 
                AND rownum <= 10000 
         GROUP  BY s_store_id), 
     csr 
     AS (SELECT cp_catalog_page_id, 
                Sum(sales_price) AS sales, 
                Sum(profit)      AS profit, 
                Sum(return_amt)  AS returns, 
                Sum(net_loss)    AS profit_loss 
         FROM   (SELECT cs_catalog_page_sk       AS page_sk, 
                        cs_sold_date_sk          AS date_sk, 
                        cs_ext_sales_price       AS sales_price, 
                        cs_net_profit            AS profit, 
                        Cast(0 AS DECIMAL(7, 2)) AS return_amt, 
                        Cast(0 AS DECIMAL(7, 2)) AS net_loss 
                 FROM   catalog_sales 
                 WHERE  rownum <= 10000 
                 UNION ALL 
                 SELECT cr_catalog_page_sk       AS page_sk, 
                        cr_returned_date_sk      AS date_sk, 
                        Cast(0 AS DECIMAL(7, 2)) AS sales_price, 
                        Cast(0 AS DECIMAL(7, 2)) AS profit, 
                        cr_return_amount         AS return_amt, 
                        cr_net_loss              AS net_loss 
                 FROM   catalog_returns 
                 WHERE  rownum <= 10000) salesreturns, 
                date_dim, 
                catalog_page 
         WHERE  date_sk = d_date_sk 
                AND d_date BETWEEN To_char(To_date('2000-08-19', 'yyyy/mm/dd'), 
                                   'yyyy-mm-dd') 
                                   AND ( 
                                       To_char(To_date('2000-08-19', 
                                               'yyyy/mm/dd') + 
                                               14, 
                                       'yyyy-mm-dd') ) 
                AND page_sk = cp_catalog_page_sk 
                AND rownum <= 10000 
         GROUP  BY cp_catalog_page_id), 
     wsr 
     AS (SELECT web_site_id, 
                Sum(sales_price) AS sales, 
                Sum(profit)      AS profit, 
                Sum(return_amt)  AS returns, 
                Sum(net_loss)    AS profit_loss 
         FROM   (SELECT ws_web_site_sk           AS wsr_web_site_sk, 
                        ws_sold_date_sk          AS date_sk, 
                        ws_ext_sales_price       AS sales_price, 
                        ws_net_profit            AS profit, 
                        Cast(0 AS DECIMAL(7, 2)) AS return_amt, 
                        Cast(0 AS DECIMAL(7, 2)) AS net_loss 
                 FROM   web_sales 
                 WHERE  rownum <= 10000 
                 UNION ALL 
                 SELECT ws_web_site_sk           AS wsr_web_site_sk, 
                        wr_returned_date_sk      AS date_sk, 
                        Cast(0 AS DECIMAL(7, 2)) AS sales_price, 
                        Cast(0 AS DECIMAL(7, 2)) AS profit, 
                        wr_return_amt            AS return_amt, 
                        wr_net_loss              AS net_loss 
                 FROM   web_returns 
                        LEFT OUTER JOIN web_sales 
                                     ON ( wr_item_sk = ws_item_sk 
                                          AND wr_order_number = ws_order_number 
                                        ) 
                 WHERE  rownum <= 10000) salesreturns, 
                date_dim, 
                web_site 
         WHERE  date_sk = d_date_sk 
                AND d_date BETWEEN To_char(To_date('2000-08-19', 'yyyy/mm/dd'), 
                                   'yyyy-mm-dd') 
                                   AND ( 
                                       To_char(To_date('2000-08-19', 
                                               'yyyy/mm/dd') + 
                                               14, 
                                       'yyyy-mm-dd') ) 
                AND wsr_web_site_sk = web_site_sk 
                AND rownum <= 10000 
         GROUP  BY web_site_id) 
SELECT * 
FROM   (SELECT channel, 
               id, 
               Sum(sales)   AS sales, 
               Sum(returns) AS returns, 
               Sum(profit)  AS profit 
        FROM   (SELECT 'store channel'          AS channel, 
                       'store' 
                       || s_store_id            AS id, 
                       sales, 
                       returns, 
                       ( profit - profit_loss ) AS profit 
                FROM   ssr
                where rownum <= 10000
                UNION ALL 
                SELECT 'catalog channel'        AS channel, 
                       'catalog_page' 
                       || cp_catalog_page_id    AS id, 
                       sales, 
                       returns, 
                       ( profit - profit_loss ) AS profit 
                FROM   csr
                where rownum <= 10000
                UNION ALL 
                SELECT 'web channel'            AS channel, 
                       'web_site' 
                       || web_site_id           AS id, 
                       sales, 
                       returns, 
                       ( profit - profit_loss ) AS profit 
                FROM   wsr
                where rownum <= 10000) x
        GROUP  BY rollup ( channel, id ) 
        ORDER  BY channel, 
                  id) 
WHERE  rownum <= 100; 