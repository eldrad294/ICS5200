SELECT *
FROM   (SELECT dt.d_year,
               item.i_brand_id     brand_id,
               item.i_brand        brand,
               Sum(ss_sales_price) sum_agg
        FROM   date_dim dt,
               store_sales,
               item
        WHERE  dt.d_date_sk between 2415522 and 2420000
               and dt.d_date_sk = store_sales.ss_sold_date_sk
               AND store_sales.ss_item_sk = item.i_item_sk
               AND item.i_manufact_id = 816
               AND dt.d_moy = 11
               and rownum <= 10000
        GROUP  BY dt.d_year,
                  item.i_brand,
                  item.i_brand_id
        ORDER  BY dt.d_year,
                  sum_agg DESC,
                  brand_id)
WHERE  rownum <= 100;
