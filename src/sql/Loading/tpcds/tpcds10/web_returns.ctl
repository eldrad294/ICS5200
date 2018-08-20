LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds10/web_returns.dat'
REPLACE
INTO TABLE tpcds10.web_returns
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
wr_returned_date_sk INTEGER EXTERNAL,
wr_returned_time_sk INTEGER EXTERNAL,
wr_item_sk      INTEGER EXTERNAL,
wr_refunded_customer_sk INTEGER EXTERNAL,
wr_refunded_cdemo_sk   INTEGER EXTERNAL,
wr_refunded_hdemo_sk  INTEGER EXTERNAL,
wr_refunded_addr_sk INTEGER EXTERNAL,
wr_returning_customer_sk INTEGER EXTERNAL,
wr_returning_cdemo_sk  INTEGER EXTERNAL,
wr_returning_hdemo_sk INTEGER EXTERNAL,
wr_returning_addr_sk   INTEGER EXTERNAL,
wr_web_page_sk    INTEGER EXTERNAL,
wr_reason_sk    INTEGER EXTERNAL,
wr_order_number   INTEGER EXTERNAL,
wr_return_quantity   INTEGER EXTERNAL,
wr_return_amt  DECIMAL EXTERNAL,
wr_return_tax   DECIMAL EXTERNAL,
wr_return_amt_inc_tax  DECIMAL EXTERNAL,
wr_fee  DECIMAL EXTERNAL,
wr_return_ship_cost  DECIMAL EXTERNAL,
wr_refunded_cash  DECIMAL EXTERNAL,
wr_reversed_charge  DECIMAL EXTERNAL,
wr_account_credit DECIMAL EXTERNAL,
wr_net_loss  DECIMAL EXTERNAL
)