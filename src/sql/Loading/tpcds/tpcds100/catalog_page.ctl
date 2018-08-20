LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds1/catalog_page.dat'
REPLACE
INTO TABLE tpcds1.catalog_page
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
cp_catalog_page_sk  INTEGER EXTERNAL,
cp_catalog_page_id  ,
cp_start_date_sk  INTEGER EXTERNAL ,
cp_end_date_sk  INTEGER EXTERNAL,
cp_department     ,
cp_catalog_number INTEGER EXTERNAL ,
cp_catalog_page_number INTEGER EXTERNAL,
cp_description    ,
cp_type
)