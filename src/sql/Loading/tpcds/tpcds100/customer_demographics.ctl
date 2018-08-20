LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds100/customer_demographics.dat'
REPLACE
INTO TABLE tpcds100.customer_demographics
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
cd_demo_sk INTEGER EXTERNAL,
cd_gender ,
cd_marital_status  ,
cd_education_status ,
cd_purchase_estimate INTEGER EXTERNAL,
cd_credit_rating ,
cd_dep_count INTEGER EXTERNAL,
cd_dep_employed_count INTEGER EXTERNAL,
cd_dep_college_count INTEGER EXTERNAL
)