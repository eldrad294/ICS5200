LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/customer_address.dat'
REPLACE
INTO TABLE tpcds1.customer_address
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(cd_demo_sk INTEGER EXTERNAL,
cd_gender,
cd_marital_status,
cd_education_status,
cd_purchase_estimate INTEGER EXTERNAL,
cd_credit_rating,
cd_dep_count INTEGER EXTERNAL,
cd_dep_employed_count,
cd_dep_college_count
)