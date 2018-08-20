LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds100/call_center.dat'
REPLACE
INTO TABLE tpcds100.call_center
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
cc_call_center_sk INTEGER EXTERNAL,
cc_call_center_id,
cc_rec_start_date ,
cc_rec_end_date,
cc_closed_date_sk INTEGER EXTERNAL,
cc_open_date_sk  INTEGER EXTERNAL,
cc_name    ,
cc_class ,
cc_employees INTEGER EXTERNAL ,
cc_sq_ft  INTEGER EXTERNAL,
cc_hours ,
cc_manager ,
cc_mkt_id INTEGER EXTERNAL,
cc_mkt_class ,
cc_mkt_desc ,
cc_market_manager ,
cc_division   INTEGER EXTERNAL ,
cc_division_name  ,
cc_company INTEGER EXTERNAL ,
cc_company_name ,
cc_street_number ,
cc_street_name,
cc_street_type ,
cc_suite_number  ,
cc_city   ,
cc_county  ,
cc_state  ,
cc_zip   ,
cc_country ,
cc_gmt_offset DECIMAL EXTERNAL ,
cc_tax_percentage DECIMAL EXTERNAL
)