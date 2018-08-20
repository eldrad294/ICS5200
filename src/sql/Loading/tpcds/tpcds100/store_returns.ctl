LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds100/store_returns.dat'
REPLACE
INTO TABLE tpcds100.store_returns
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
sr_returned_date_sk INTEGER EXTERNAL,
sr_return_time_sk  INTEGER EXTERNAL,
sr_item_sk      INTEGER EXTERNAL,
sr_customer_sk   INTEGER EXTERNAL,
sr_cdemo_sk    INTEGER EXTERNAL,
sr_hdemo_sk   INTEGER EXTERNAL,
sr_addr_sk   INTEGER EXTERNAL,
sr_store_sk   INTEGER EXTERNAL,
sr_reason_sk   INTEGER EXTERNAL,
sr_ticket_number    INTEGER EXTERNAL,
sr_return_quantity  INTEGER EXTERNAL,
sr_return_amt    DECIMAL EXTERNAL,
sr_return_tax DECIMAL EXTERNAL,
sr_return_amt_inc_tax DECIMAL EXTERNAL,
sr_fee  DECIMAL EXTERNAL,
sr_return_ship_cost DECIMAL EXTERNAL,
sr_refunded_cash  DECIMAL EXTERNAL,
sr_reversed_charge   DECIMAL EXTERNAL,
sr_store_credit  DECIMAL EXTERNAL,
sr_net_loss   DECIMAL EXTERNAL
)