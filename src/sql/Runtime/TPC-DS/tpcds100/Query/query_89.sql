SELECT *
FROM   (SELECT *
        FROM  (SELECT i_category,
                      i_class,
                      i_brand,
                      s_store_name,
                      s_company_name,
                      d_moy,
                      Sum(ss_sales_price) sum_sales,
                      Avg(Sum(ss_sales_price))
                        OVER (
                          partition BY i_category, i_brand, s_store_name,
                        s_company_name)
                                          avg_monthly_sales
               FROM   item,
                      store_sales,
                      date_dim,
                      store
               WHERE  ss_item_sk = i_item_sk
                      AND ss_sold_date_sk = d_date_sk
                      and d_date_sk between 2415522 and 2425522
                      AND ss_store_sk = s_store_sk
                      AND d_year IN ( 1998 )
                      AND ( ( i_category IN ( 'Music', 'Sports', 'Women' )
                              AND i_class IN ( 'pop', 'athletic shoes',
                                               'dresses' ) )
                             OR ( i_category IN ( 'Children', 'Men', 'Jewelry' )
                                  AND i_class IN ( 'toddlers', 'shirts',
                                                   'pendants' ) )
                          )
                      and rownum <= 10000
               GROUP  BY i_category,
                         i_class,
                         i_brand,
                         s_store_name,
                         s_company_name,
                         d_moy) tmp1
        WHERE  CASE
                 WHEN ( avg_monthly_sales <> 0 ) THEN (
                 Abs(sum_sales - avg_monthly_sales) / avg_monthly_sales )
                 ELSE NULL
               END > 0.1
        and rownum <= 10000
        ORDER  BY sum_sales - avg_monthly_sales,
                  s_store_name)
WHERE  rownum <= 100;