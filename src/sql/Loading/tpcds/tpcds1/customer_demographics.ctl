LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/customer_demographics.dat'
REPLACE
INTO TABLE tpcds1.customer_demographics
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(ca_address_sk INTEGER EXTERNAL,
ca_address_id,
ca_street_number,
ca_street_name,
ca_street_type,
ca_suite_number,
ca_city,
ca_county,
ca_state,
ca_zip,
ca_country,
ca_gmt_offset DECIMAL EXTERNAL,
ca_location_type)