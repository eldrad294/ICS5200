LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/time_dim.dat'
REPLACE
INTO TABLE tpcds50.time_dim
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
t_time_sk  INTEGER EXTERNAL,
t_time_id ,
t_time INTEGER EXTERNAL ,
t_hour INTEGER EXTERNAL ,
t_minute INTEGER EXTERNAL,
t_second  INTEGER EXTERNAL ,
t_am_pm,
t_shift ,
t_sub_shift ,
t_meal_time
)