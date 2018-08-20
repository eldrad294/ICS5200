LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/web_sales.dat'
REPLACE
INTO TABLE tpcds1.web_sales
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
ws_sold_date_sk INTEGER EXTERNAL,
ws_sold_time_sk INTEGER EXTERNAL,
ws_ship_date_sk  INTEGER EXTERNAL,
ws_item_sk     INTEGER EXTERNAL,
ws_bill_customer_sk   INTEGER EXTERNAL,
ws_bill_cdemo_sk    INTEGER EXTERNAL,
ws_bill_hdemo_sk INTEGER EXTERNAL,
ws_bill_addr_sk    INTEGER EXTERNAL,
ws_ship_customer_sk    INTEGER EXTERNAL,
ws_ship_cdemo_sk INTEGER EXTERNAL,
ws_ship_hdemo_sk  INTEGER EXTERNAL,
ws_ship_addr_sk  INTEGER EXTERNAL,
ws_web_page_sk   INTEGER EXTERNAL,
ws_web_site_sk  INTEGER EXTERNAL,
ws_ship_mode_sk  INTEGER EXTERNAL,
ws_warehouse_sk   INTEGER EXTERNAL,
ws_promo_sk     INTEGER EXTERNAL,
ws_order_number   INTEGER EXTERNAL,
ws_quantity     INTEGER EXTERNAL,
ws_wholesale_cost DECIMAL EXTERNAL,
ws_list_price    DECIMAL EXTERNAL,
ws_sales_price   DECIMAL EXTERNAL,
ws_ext_discount_amt DECIMAL EXTERNAL,
ws_ext_sales_price   DECIMAL EXTERNAL,
ws_ext_wholesale_cost   DECIMAL EXTERNAL,
ws_ext_list_price  DECIMAL EXTERNAL,
ws_ext_tax      DECIMAL EXTERNAL,
ws_coupon_amt    DECIMAL EXTERNAL,
ws_ext_ship_cost DECIMAL EXTERNAL,
ws_net_paid DECIMAL EXTERNAL,
ws_net_paid_inc_tax DECIMAL EXTERNAL,
ws_net_paid_inc_ship DECIMAL EXTERNAL,
ws_net_paid_inc_ship_tax DECIMAL EXTERNAL,
ws_net_profit DECIMAL EXTERNAL
)