drop table cc2_catalog_returns;
create table cc2_catalog_returns as ( select 6 col1, CR_CATALOG_PAGE_SK, 6 col2, CR_ITEM_SK, CR_NET_LOSS, CR_ORDER_NUMBER, CR_REASON_SK, CR_REFUNDED_ADDR_SK, CR_REFUNDED_CASH, CR_REFUNDED_CDEMO_SK, CR_REFUNDED_CUSTOMER_SK, CR_REFUNDED_HDEMO_SK, CR_RETURING_ADDR_SK, CR_RETURNING_CDEMO_SK, CR_RETURNING_CUSTOMER_SK, CR_RETURNING_HDEMO_SK, 6 col3, CR_RETURN_AMT_INC_TAX, CR_RETURN_DATE_SK, CR_RETURN_QUANTITY, CR_RETURN_SHIP_COST, CR_RETURN_TAX, CR_RETURN_TIME_SK, CR_REVERSED_CHARDE, CR_SHIP_DATE_SK, CR_SHIP_MODE_SK, CR_WAREHOUSE_SK from crv);
rollback;