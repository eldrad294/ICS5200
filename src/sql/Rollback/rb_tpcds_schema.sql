update tpcds1.MON_KILL_LONG_RUNNING set running=0;
update tpcds10.MON_KILL_LONG_RUNNING set running=0;
update tpcds100.MON_KILL_LONG_RUNNING set running=0;
commit;
drop table CATV;
drop table CCV;
drop table CRV;
drop table CUSTV;
drop table CC2_CALL_CENTER;
drop table CC2_STORE_RETURNS;
drop table CC2_CATALOG_RETURNS;
drop table ITEMV;
drop table PROMV;
drop table SRV;
drop table STORV;
drop table S_CALL_CENTER;
drop table S_CATALOG_PAGE;
drop table S_CUSTOMER_M;
drop table S_INVENTORY;
drop table S_ITEM;
drop table S_PROMOTION;
drop table S_STORE;
drop table S_WAREHOUSE;
drop table S_WEB_PAGE;
drop table S_WEB_SITE;
drop table WEBSV;
drop table WEBV;
drop table WRHSV;
drop table WRV;
drop table CALL_CENTER;
drop table CATALOG_PAGE;
drop table CATALOG_RETURNS;
drop table CATALOG_SALES;
drop table CUSTOMER;
drop table CUSTOMER_ADDRESS;
drop table CUSTOMER_DEMOGRAPHICS;
drop table DATE_DIM;
drop table DBGEN_VERSION;
drop table HOUSEHOLD_DEMOGRAPHICS;
drop table INCOME_BAND;
drop table INVENTORY;
drop table ITEM;
drop table PROMOTION;
drop table REASON;
drop table SHIP_MODE;
drop table STORE;
drop table STORE_RETURNS;
drop table STORE_SALES;
drop table TIME_DIM;
drop table WAREHOUSE;
drop table WEB_PAGE;
drop table WEB_RETURNS;
drop table WEB_SALES;
drop table WEB_SITE;
drop table REP_EXECUTION_PLANS;
drop table REP_EXPLAIN_PLANS;
drop table REP_TPC_DESCRIBE;
drop table MON_KILL_LONG_RUNNING;
drop procedure kill_long_running;