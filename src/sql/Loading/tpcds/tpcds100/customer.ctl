LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds100/customer.dat'
REPLACE
INTO TABLE tpcds100.customer
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
c_customer_sk INTEGER EXTERNAL ,
c_customer_id ,
c_current_cdemo_sk INTEGER EXTERNAL,
c_current_hdemo_sk INTEGER EXTERNAL,
c_current_addr_sk INTEGER EXTERNAL,
c_first_shipto_date_sk INTEGER EXTERNAL,
c_first_sales_date_sk INTEGER EXTERNAL,
c_salutation ,
c_first_name,
c_last_name ,
c_preferred_cust_flag ,
c_birth_day   INTEGER EXTERNAL,
c_birth_month INTEGER EXTERNAL,
c_birth_year   INTEGER EXTERNAL,
c_birth_country  ,
c_login       ,
c_email_address DECIMAL EXTERNAL,
c_last_review_date DECIMAL EXTERNAL
)