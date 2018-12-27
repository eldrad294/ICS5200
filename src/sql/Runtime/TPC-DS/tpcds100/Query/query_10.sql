SELECT *
FROM   (SELECT cd_gender,
               cd_marital_status,
               cd_education_status,
               Count(*) cnt1,
               cd_purchase_estimate,
               Count(*) cnt2,
               cd_credit_rating,
               Count(*) cnt3,
               cd_dep_count,
               Count(*) cnt4,
               cd_dep_employed_count,
               Count(*) cnt5,
               cd_dep_college_count,
               Count(*) cnt6
        FROM   customer c,
               customer_address ca,
               customer_demographics
        WHERE  c.c_current_addr_sk = ca.ca_address_sk
               and c_customer_sk between 579 and 700
               AND ca_county IN ( 'Storey County', 'Marquette County',
                                  'Warren County',
                                  'Cochran County',
                                                   'Kandiyohi County' )
               and rownum <= 10000
               AND cd_demo_sk = c.c_current_cdemo_sk
               AND EXISTS (SELECT *
                           FROM   store_sales,
                                  date_dim
                           WHERE  c.c_customer_sk = ss_customer_sk
                                  AND ss_sold_date_sk = d_date_sk
                                  AND d_year = 2001
                                  AND d_moy BETWEEN 1 AND 1 + 3)
                                  and rownum <= 1
               AND ( EXISTS (SELECT *
                             FROM   web_sales,
                                    date_dim
                             WHERE  c.c_customer_sk = ws_bill_customer_sk
                                    AND ws_sold_date_sk = d_date_sk
                                    AND d_year = 2001
                                    AND d_moy BETWEEN 1 AND 1 + 3
                                    and rownum <= 1 )
                      OR EXISTS (SELECT *
                                 FROM   catalog_sales,
                                        date_dim
                                 WHERE  c.c_customer_sk = cs_ship_customer_sk
                                        AND cs_sold_date_sk = d_date_sk
                                        AND d_year = 2001
                                        AND d_moy BETWEEN 1 AND 1 + 3
                                        and rownum <= 1 ) )
        GROUP  BY cd_gender,
                  cd_marital_status,
                  cd_education_status,
                  cd_purchase_estimate,
                  cd_credit_rating,
                  cd_dep_count,
                  cd_dep_employed_count,
                  cd_dep_college_count
        ORDER  BY cd_gender,
                  cd_marital_status,
                  cd_education_status,
                  cd_purchase_estimate,
                  cd_credit_rating,
                  cd_dep_count,
                  cd_dep_employed_count,
                  cd_dep_college_count)
WHERE  rownum <= 100;