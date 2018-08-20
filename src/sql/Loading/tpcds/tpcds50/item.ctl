LOAD DATA
INFILE '/mnt/raid5/DataGeneration_ICS5200/TPC-DS/tpcds50/item.dat'
REPLACE
INTO TABLE tpcds50.item
FIELDS TERMINATED BY '|' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
i_item_sk INTEGER EXTERNAL,
i_item_id ,
i_rec_start_date,
i_rec_end_date ,
i_item_desc ,
i_current_price DECIMAL EXTERNAL,
i_wholesale_cost DECIMAL EXTERNAL,
i_brand_id INTEGER EXTERNAL,
i_brand ,
i_class_id INTEGER EXTERNAL,
i_class ,
i_category_id INTEGER EXTERNAL,
i_category ,
i_manufact_id INTEGER EXTERNAL,
i_manufact ,
i_size ,
i_formulation ,
i_color ,
i_units,
i_container ,
i_manager_id  INTEGER EXTERNAL,
i_product_name
)