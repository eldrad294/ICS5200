WITH customer_total_return
     AS (SELECT sr_customer_sk AS ctr_customer_sk,
                sr_store_sk    AS ctr_store_sk,
                Sum(sr_fee)    AS ctr_total_return
         FROM   store_returns,
                date_dim
         WHERE  sr_returned_date_sk = d_date_sk
                AND d_year = 2000
                and sr_item_sk = 99700
                and sr_ticket_number = 140324
                AND rownum <= 10000
         GROUP  BY sr_customer_sk,
                   sr_store_sk)
SELECT *
FROM   (SELECT c_customer_id
        FROM   customer_total_return ctr1,
               store,
               customer
        WHERE  ctr1.ctr_total_return > (SELECT Avg(ctr_total_return) * 1.2
                                        FROM   customer_total_return ctr2
                                        WHERE
               ctr1.ctr_store_sk = ctr2.ctr_store_sk)
               AND s_store_sk = ctr1.ctr_store_sk
               AND s_state = 'SD'
               AND ctr1.ctr_customer_sk = c_customer_sk
               AND rownum <= 10000
        ORDER  BY c_customer_id)
WHERE  rownum <= 100;