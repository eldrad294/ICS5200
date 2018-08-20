LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds10/promotion.dat'
REPLACE
INTO TABLE tpcds10.promotion
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
p_promo_sk INTEGER EXTERNAL,
p_promo_id    ,
p_start_date_sk  INTEGER EXTERNAL ,
p_end_date_sk INTEGER EXTERNAL ,
p_item_sk   INTEGER EXTERNAL ,
p_cost   DECIMAL EXTERNAL  ,
p_response_target INTEGER EXTERNAL,
p_promo_name    ,
p_channel_dmail  ,
p_channel_email  ,
p_channel_catalog ,
p_channel_tv    ,
p_channel_radio   ,
p_channel_press   ,
p_channel_event   ,
p_channel_demo    ,
p_channel_details ,
p_purpose       ,
p_discount_active
)