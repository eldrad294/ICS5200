LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/dbgen_version.dat'
REPLACE
INTO TABLE tpcds1.dbgen_version
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(dv_version,
dv_create_date,
dv_create_time,
dv_cmdline_args)