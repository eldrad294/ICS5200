LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds100/store_sales.dat'
REPLACE
INTO TABLE tpcds100.store_sales
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
ss_sold_date_sk INTEGER EXTERNAL,
ss_sold_time_sk INTEGER EXTERNAL,
ss_item_sk   INTEGER EXTERNAL,
ss_customer_sk   INTEGER EXTERNAL,
ss_cdemo_sk    INTEGER EXTERNAL,
ss_hdemo_sk  INTEGER EXTERNAL,
ss_addr_sk  INTEGER EXTERNAL,
ss_store_sk  INTEGER EXTERNAL,
ss_promo_sk   INTEGER EXTERNAL,
ss_ticket_number  INTEGER EXTERNAL,
ss_quantity   INTEGER EXTERNAL,
ss_wholesale_cost DECIMAL EXTERNAL,
ss_list_price  DECIMAL EXTERNAL,
ss_sales_price   DECIMAL EXTERNAL,
ss_ext_discount_amt  DECIMAL EXTERNAL,
ss_ext_sales_price DECIMAL EXTERNAL,
ss_ext_wholesale_cost DECIMAL EXTERNAL,
ss_ext_list_price  DECIMAL EXTERNAL,
ss_ext_tax     DECIMAL EXTERNAL,
ss_coupon_amt  DECIMAL EXTERNAL,
ss_net_paid    DECIMAL EXTERNAL,
ss_net_paid_inc_tax DECIMAL EXTERNAL,
ss_net_profit  DECIMAL EXTERNAL
)