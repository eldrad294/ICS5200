LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/inventory.dat'
REPLACE
INTO TABLE tpcds50.inventory
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
inv_date_sk  INTEGER EXTERNAL,
inv_item_sk   INTEGER EXTERNAL  ,
inv_warehouse_sk INTEGER EXTERNAL ,
inv_quantity_on_hand INTEGER EXTERNAL
)