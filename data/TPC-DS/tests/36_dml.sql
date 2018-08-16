update web_page set wp_rec_end_date = sysdate where wp_web_page_id in (select wp_web_page_id from webv) and wp_rec_end_date is NULL;
insert into web_page (select WP_WEB_PAGE_SK,
							WP_WEB_PAGE_ID,
							WP_REC_START_DATE,
							WP_REC_END_DATE,
							WP_CREATION_DATE_SK,
							WP_ACCESS_DATE_SK,
							WP_AUTOGEN_FLAG,
							'CUSTOMER_SK_DUMMY_VALUE',
							WP_URL,
							WP_TYPE,
							WP_CHAR_COUNT,
							WP_LINK_COUNT,
							WP_IMAGE_COUNT,
							WP_MAX_AD_COUNT
							 from webv);
commit;