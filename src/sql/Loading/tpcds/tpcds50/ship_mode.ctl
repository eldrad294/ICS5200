LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/ship_mode.dat'
REPLACE
INTO TABLE tpcds50.ship_mode
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
sm_ship_mode_sk integer external,
sm_ship_mode_id,
sm_type,
sm_code,
sm_carrier,
sm_contract
)