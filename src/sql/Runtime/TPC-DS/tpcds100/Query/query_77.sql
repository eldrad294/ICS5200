WITH ss
     AS (SELECT s_store_sk,
                Sum(ss_ext_sales_price) AS sales,
                Sum(ss_net_profit)      AS profit
         FROM   store_sales,
                date_dim,
                store
         WHERE  ss_sold_date_sk = d_date_sk
                AND d_date BETWEEN To_char(To_date('2000-08-23', 'yyyy/mm/dd'),
                                   'yyyy-mm-dd')
                                   AND (
                                       To_char(To_date('2000-08-23',
                                               'yyyy/mm/dd') +
                                               30,
                                       'yyyy-mm-dd') )
                AND ss_store_sk = s_store_sk
                and d_date_sk between 2410000 and 2420000
                and rownum <= 10000
         GROUP  BY s_store_sk),
     sr
     AS (SELECT s_store_sk,
                Sum(sr_return_amt) AS returns,
                Sum(sr_net_loss)   AS profit_loss
         FROM   store_returns,
                date_dim,
                store
         WHERE  sr_returned_date_sk = d_date_sk
                AND d_date BETWEEN To_char(To_date('2000-08-23', 'yyyy/mm/dd'),
                                   'yyyy-mm-dd')
                                   AND (
                                       To_char(To_date('2000-08-23',
                                               'yyyy/mm/dd') +
                                               30,
                                       'yyyy-mm-dd') )
                AND sr_store_sk = s_store_sk
                and d_date_sk between 2410000 and 2420000
                and rownum <= 10000
         GROUP  BY s_store_sk),
     cs
     AS (SELECT cs_call_center_sk,
                Sum(cs_ext_sales_price) AS sales,
                Sum(cs_net_profit)      AS profit
         FROM   catalog_sales,
                date_dim
         WHERE  cs_sold_date_sk = d_date_sk
                AND d_date BETWEEN To_char(To_date('2000-08-23', 'yyyy/mm/dd'),
                                   'yyyy-mm-dd') AND (
                                       To_char(To_date('2000-08-23',
                                               'yyyy/mm/dd') +
                                               30,
                                       'yyyy-mm-dd') )
                and d_date_sk between 2410000 and 2420000
                and rownum <= 10000
         GROUP  BY cs_call_center_sk),
     cr
     AS (SELECT cr_call_center_sk,
                Sum(cr_return_amount) AS returns,
                Sum(cr_net_loss)      AS profit_loss
         FROM   catalog_returns,
                date_dim
         WHERE  cr_returned_date_sk = d_date_sk
                AND d_date BETWEEN To_char(To_date('2000-08-23', 'yyyy/mm/dd'),
                                   'yyyy-mm-dd') AND (
                                       To_char(To_date('2000-08-23',
                                               'yyyy/mm/dd') +
                                               30,
                                       'yyyy-mm-dd') )
                                       and d_date_sk between 2415522 and 2425522
                and rownum <= 10000
         GROUP  BY cr_call_center_sk),
     ws
     AS (SELECT wp_web_page_sk,
                Sum(ws_ext_sales_price) AS sales,
                Sum(ws_net_profit)      AS profit
         FROM   web_sales,
                date_dim,
                web_page
         WHERE  ws_sold_date_sk = d_date_sk
                AND d_date BETWEEN To_char(To_date('2000-08-23', 'yyyy-mm-dd'),
                                   'yyyy-mm-dd')
                                   AND (
                                       To_char(To_date('2000-08-23',
                                               'yyyy-mm-dd') +
                                               30,
                                       'yyyy-mm-dd') )
                AND ws_web_page_sk = wp_web_page_sk
                and d_date_sk between 2415522 and 2425522
                and rownum <= 10000
         GROUP  BY wp_web_page_sk),
     wr
     AS (SELECT wp_web_page_sk,
                Sum(wr_return_amt) AS returns,
                Sum(wr_net_loss)   AS profit_loss
         FROM   web_returns,
                date_dim,
                web_page
         WHERE  wr_returned_date_sk = d_date_sk
                and d_date_sk between 2415522 and 2425522
                AND d_date BETWEEN To_char(To_date('2000-08-23', 'yyyy/mm/dd'),
                                   'yyyy-mm-dd')
                                   AND (
                                       To_char(To_date('2000-08-23',
                                               'yyyy/mm/dd') +
                                               30,
                                       'yyyy-mm-dd') )
                AND wr_web_page_sk = wp_web_page_sk
                and d_date_sk between 2410000 and 2420000
                and rownum <= 10000
         GROUP  BY wp_web_page_sk)
SELECT *
FROM   (SELECT channel,
               id,
               Sum(sales)   AS sales,
               Sum(returns) AS returns,
               Sum(profit)  AS profit
        FROM   (SELECT 'store channel'                       AS channel,
                       ss.s_store_sk                         AS id,
                       sales,
                       COALESCE(returns, 0)                  AS returns,
                       ( profit - COALESCE(profit_loss, 0) ) AS profit
                FROM   ss
                       LEFT JOIN sr
                              ON ss.s_store_sk = sr.s_store_sk
                where rownum <= 10000
                UNION ALL
                SELECT 'catalog channel'        AS channel,
                       cs_call_center_sk        AS id,
                       sales,
                       returns,
                       ( profit - profit_loss ) AS profit
                FROM   cs,
                       cr
                where rownum <= 10000
                UNION ALL
                SELECT 'web channel'                         AS channel,
                       ws.wp_web_page_sk                     AS id,
                       sales,
                       COALESCE(returns, 0)                  returns,
                       ( profit - COALESCE(profit_loss, 0) ) AS profit
                FROM   ws
                       LEFT JOIN wr
                              ON ws.wp_web_page_sk = wr.wp_web_page_sk) x
                where rownum <= 10000
        GROUP  BY rollup ( channel, id )
        ORDER  BY channel,
                  id)
WHERE  rownum <= 100;