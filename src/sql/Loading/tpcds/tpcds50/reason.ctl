LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/reason.dat'
REPLACE
INTO TABLE tpcds50.reason
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
r_reason_sk INTEGER EXTERNAL,
r_reason_id ,
r_reason_desc
)