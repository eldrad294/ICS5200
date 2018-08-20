LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds10/customer_address.dat'
REPLACE
INTO TABLE tpcds10.customer_address
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
ca_address_sk INTEGER EXTERNAL,
ca_address_id ,
ca_street_number,
ca_street_name ,
ca_street_type ,
ca_suite_number ,
ca_city ,
ca_county ,
ca_state  ,
ca_zip    ,
ca_country   ,
ca_gmt_offset DECIMAL EXTERNAL,
ca_location_type)