LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/catalog_returns.dat'
REPLACE
INTO TABLE tpcds50.catalog_returns
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
cr_returned_date_sk INTEGER EXTERNAL,
cr_returned_time_sk  INTEGER EXTERNAL,
cr_item_sk   INTEGER EXTERNAL,
cr_refunded_customer_sk INTEGER EXTERNAL,
cr_refunded_cdemo_sk    INTEGER EXTERNAL,
cr_refunded_hdemo_sk   INTEGER EXTERNAL,
cr_refunded_addr_sk    INTEGER EXTERNAL,
cr_returning_customer_sk INTEGER EXTERNAL,
cr_returning_cdemo_sk  INTEGER EXTERNAL,
cr_returning_hdemo_sk  INTEGER EXTERNAL,
cr_returning_addr_sk   INTEGER EXTERNAL,
cr_call_center_sk  INTEGER EXTERNAL,
cr_catalog_page_sk  INTEGER EXTERNAL,
cr_ship_mode_sk   INTEGER EXTERNAL,
cr_warehouse_sk   INTEGER EXTERNAL,
cr_reason_sk    INTEGER EXTERNAL,
cr_order_number   INTEGER EXTERNAL,
cr_return_quantity  INTEGER EXTERNAL,
cr_return_amount  DECIMAL EXTERNAL,
cr_return_tax      DECIMAL EXTERNAL,
cr_return_amt_inc_tax DECIMAL EXTERNAL,
cr_fee         DECIMAL EXTERNAL,
cr_return_ship_cost DECIMAL EXTERNAL,
cr_refunded_cash  DECIMAL EXTERNAL,
cr_reversed_charge  DECIMAL EXTERNAL,
cr_store_credit   DECIMAL EXTERNAL,
cr_net_loss    DECIMAL EXTERNAL
)