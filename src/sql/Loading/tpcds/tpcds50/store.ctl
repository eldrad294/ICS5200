LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/store.dat'
REPLACE
INTO TABLE tpcds50.store
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
s_store_sk INTEGER EXTERNAL,
s_store_id  ,
s_rec_start_date  ,
s_rec_end_date  ,
s_closed_date_sk INTEGER EXTERNAL ,
s_store_name  ,
s_number_employees INTEGER EXTERNAL,
s_floor_space INTEGER EXTERNAL,
s_hours  ,
s_manager  ,
s_market_id INTEGER EXTERNAL ,
s_geography_class ,
s_market_desc   ,
s_market_manager  ,
s_division_id  INTEGER EXTERNAL,
s_division_name ,
s_company_id  INTEGER EXTERNAL ,
s_company_name ,
s_street_number  ,
s_street_name,
s_street_type  ,
s_suite_number  ,
s_city     ,
s_county  ,
s_state  ,
s_zip    ,
s_country    ,
s_gmt_offset DECIMAL EXTERNAL,
s_tax_precentage DECIMAL EXTERNAL
)