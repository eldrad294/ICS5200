select * from (select  count(*) from (
    select distinct c_last_name, c_first_name, d_date
    from store_sales, date_dim, customer
          where store_sales.ss_sold_date_sk = date_dim.d_date_sk
      and c_customer_sk between 579 and 900
      and store_sales.ss_customer_sk = customer.c_customer_sk
      and d_month_seq between 1183 and 1183 + 11
      and rownum <= 10000
  intersect
    select distinct c_last_name, c_first_name, d_date
    from catalog_sales, date_dim, customer
          where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
          and c_customer_sk between 579 and 900
      and catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
      and d_month_seq between 1183 and 1183 + 11
      and rownum <= 10000
  intersect
    select distinct c_last_name, c_first_name, d_date
    from web_sales, date_dim, customer
          where web_sales.ws_sold_date_sk = date_dim.d_date_sk
          and c_customer_sk between 579 and 900
      and web_sales.ws_bill_customer_sk = customer.c_customer_sk
      and d_month_seq between 1183 and 1183 + 11
      and rownum <= 10000
) hot_cust
 ) where rownum <= 100;