drop table webv;
create table webv tablespace tpcds_benchmark as
select web_page_seq.nextVal wp_web_page_sk ,wpag_web_page_id wp_web_page_id ,sysdate wp_rec_start_date ,cast(null as date) wp_rec_end_date ,d1.d_date_sk wp_creation_date_sk ,d2.d_date_sk wp_access_date_sk ,wpag_autogen_flag wp_autogen_flag ,wpag_url wp_url ,wpag_type wp_type ,WPAG_CHAR_COUNT wp_char_count ,WPAG_LINK_COUNT wp_link_count ,WPAG_IMAGE_COUNT wp_image_count ,WPAG_MAX_AD_COUNT wp_max_ad_count
from s_web_page
left outer join date_dim d1 on WPAG_CREATE_DATE = d1.d_date
left outer join date_dim d2 on wpag_access_date = d2.d_date;
select count(*) from s_web_page;
select count(*) from webv;
