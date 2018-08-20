LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/household_demographics.dat'
REPLACE
INTO TABLE tpcds50.household_demographics
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
hd_demo_sk INTEGER EXTERNAL,
hd_income_band_sk INTEGER EXTERNAL,
hd_buy_potential ,
hd_dep_count   INTEGER EXTERNAL,
hd_vehicle_count INTEGER EXTERNAL
)