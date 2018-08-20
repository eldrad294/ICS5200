LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/catalog_sales.dat'
REPLACE
INTO TABLE tpcds1.catalog_sales
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
cs_sold_date_sk INTEGER EXTERNAL,
cs_sold_time_sk INTEGER EXTERNAL,
cs_ship_date_sk INTEGER EXTERNAL,
cs_bill_customer_sk INTEGER EXTERNAL,
cs_bill_cdemo_sk  INTEGER EXTERNAL,
cs_bill_hdemo_sk  INTEGER EXTERNAL,
cs_bill_addr_sk   INTEGER EXTERNAL,
cs_ship_customer_sk  INTEGER EXTERNAL,
cs_ship_cdemo_sk   INTEGER EXTERNAL,
cs_ship_hdemo_sk  INTEGER EXTERNAL,
cs_ship_addr_sk    INTEGER EXTERNAL,
cs_call_center_sk INTEGER EXTERNAL,
cs_catalog_page_sk  INTEGER EXTERNAL,
cs_ship_mode_sk   INTEGER EXTERNAL,
cs_warehouse_sk  INTEGER EXTERNAL,
cs_item_sk  INTEGER EXTERNAL,
cs_promo_sk INTEGER EXTERNAL,
cs_order_number  INTEGER EXTERNAL,
cs_quantity  INTEGER EXTERNAL,
cs_wholesale_cost DECIMAL EXTERNAL,
cs_list_price DECIMAL EXTERNAL,
cs_sales_price DECIMAL EXTERNAL,
cs_ext_discount_amt  DECIMAL EXTERNAL,
cs_ext_sales_price DECIMAL EXTERNAL,
cs_ext_wholesale_cost DECIMAL EXTERNAL,
cs_ext_list_price DECIMAL EXTERNAL,
cs_ext_tax  DECIMAL EXTERNAL,
cs_coupon_amt  DECIMAL EXTERNAL,
cs_ext_ship_cost DECIMAL EXTERNAL,
cs_net_paid DECIMAL EXTERNAL,
cs_net_paid_inc_tax DECIMAL EXTERNAL,
cs_net_paid_inc_ship  DECIMAL EXTERNAL,
cs_net_paid_inc_ship_tax DECIMAL EXTERNAL,
cs_net_profit  DECIMAL EXTERNAL
)