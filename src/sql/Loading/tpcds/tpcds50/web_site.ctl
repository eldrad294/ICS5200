LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/web_site.dat'
REPLACE
INTO TABLE tpcds1.web_site
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
web_site_sk INTEGER EXTERNAL,
web_site_id,
web_rec_start_date,
web_rec_end_date,
web_name   ,
web_open_date_sk  INTEGER EXTERNAL,
web_close_date_sk INTEGER EXTERNAL,
web_class ,
web_manager ,
web_mkt_id INTEGER EXTERNAL,
web_mkt_class  ,
web_mkt_desc   ,
web_market_manager,
web_company_id  INTEGER EXTERNAL ,
web_company_name ,
web_street_number ,
web_street_name ,
web_street_type   ,
web_suite_number  ,
web_city     ,
web_county ,
web_state  ,
web_zip   ,
web_country  ,
web_gmt_offset DECIMAL EXTERNAL ,
web_tax_percentage DECIMAL EXTERNAL
)