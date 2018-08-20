LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/web_page.dat'
REPLACE
INTO TABLE tpcds50.web_page
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
wp_web_page_sk INTEGER EXTERNAL,
wp_web_page_id   ,
wp_rec_start_date  ,
wp_rec_end_date  ,
wp_creation_date_sk INTEGER EXTERNAL,
wp_access_date_sk  INTEGER EXTERNAL,
wp_autogen_flag  ,
wp_customer_sk  INTEGER EXTERNAL ,
wp_url     ,
wp_type    ,
wp_char_count  INTEGER EXTERNAL ,
wp_link_count  INTEGER EXTERNAL ,
wp_image_count  INTEGER EXTERNAL  ,
wp_max_ad_count  INTEGER EXTERNAL
)