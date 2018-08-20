LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds10/date_dim.dat'
REPLACE
INTO TABLE tpcds10.date_dim
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(d_date_sk INTEGER EXTERNAL,
d_date_id ,
d_date,
d_month_seq INTEGER EXTERNAL,
d_week_seq INTEGER EXTERNAL,
d_quarter_seq INTEGER EXTERNAL,
d_year INTEGER EXTERNAL,
d_dow INTEGER EXTERNAL,
d_moy INTEGER EXTERNAL,
d_dom INTEGER EXTERNAL,
d_qoy INTEGER EXTERNAL,
d_fy_year INTEGER EXTERNAL,
d_fy_quarter_seq  INTEGER EXTERNAL,
d_fy_week_seq INTEGER EXTERNAL,
d_day_name   ,
d_quarter_name     ,
d_holiday     ,
d_weekend  ,
d_following_holiday,
d_first_dom  INTEGER EXTERNAL,
d_last_dom INTEGER EXTERNAL,
d_same_day_ly    INTEGER EXTERNAL,
d_same_day_lq INTEGER EXTERNAL,
d_current_day  ,
d_current_week  ,
d_current_month ,
d_current_quarter ,
d_current_year
)