LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/income_band.dat'
REPLACE
INTO TABLE tpcds50.income_band
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
ib_income_band_sk INTEGER EXTERNAL,
ib_lower_bound INTEGER EXTERNAL,
ib_upper_bound INTEGER EXTERNAL
)