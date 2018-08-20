LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/warehouse.dat'
REPLACE
INTO TABLE tpcds50.warehouse
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
w_warehouse_sk INTEGER EXTERNAL,
w_warehouse_id ,
w_warehouse_name ,
w_warehouse_sq_ft  ,
w_street_number,
w_street_name ,
w_street_type ,
w_suite_number ,
w_city   ,
w_county  ,
w_state ,
w_zip   ,
w_country  ,
w_gmt_offset DECIMAL EXTERNAL
)