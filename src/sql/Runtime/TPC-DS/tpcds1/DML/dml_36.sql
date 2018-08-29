drop table crv;
update catalog_returns set CR_RETURNED_DATE_SK = null where CR_RETURNED_DATE_SK like '%-47%';
create table crv tablespace tpcds_benchmark as
select d_date_sk cr_return_date_sk ,
t_time_sk cr_return_time_sk ,
0 CR_SHIP_DATE_SK ,
i_item_sk cr_item_sk ,
c1.c_customer_sk cr_refunded_customer_sk ,
c1.c_current_cdemo_sk cr_refunded_cdemo_sk ,
c1.c_current_hdemo_sk cr_refunded_hdemo_sk ,
c1.c_current_addr_sk cr_refunded_addr_sk ,
c2.c_customer_sk cr_returning_customer_sk ,
c2.c_current_cdemo_sk cr_returning_cdemo_sk ,
c2.c_current_hdemo_sk cr_returning_hdemo_sk ,
c2.c_current_addr_sk cr_returing_addr_sk ,
--cc_call_center_sk cr_call_center_sk ,
0 CR_CATALOG_PAGE_SK ,
0 CR_SHIP_MODE_SK ,
0 CR_WAREHOUSE_SK ,
r_reason_sk cr_reason_sk ,
CR_ORDER_NUMBER cr_order_number ,
CR_RETURN_QUANTITY cr_return_quantity ,
CR_RETURN_AMOUNT cr_return_amt ,
CR_RETURN_TAX cr_return_tax ,
CR_RETURN_AMOUNT + CR_RETURN_TAX as cr_return_amt_inc_tax ,
CR_RETURN_TAX cr_fee ,
CR_RETURN_SHIP_COST CR_RETURN_SHIP_COST ,
CR_REFUNDED_CASH cr_refunded_cash ,
CR_REVERSED_CHARGE cr_reversed_charde ,
CR_STORE_CREDIT cr_merchant_credit ,
CR_RETURN_AMOUNT+CR_RETURN_TAX+CR_RETURN_TAX -CR_REFUNDED_CASH-CR_REVERSED_CHARGE-CR_STORE_CREDIT cr_net_loss
from catalog_returns
left outer join date_dim on (to_char(CR_RETURNED_DATE_SK) = d_date)
left outer join time_dim on (( cast(substr(CR_RETURNED_TIME_SK,1,2) as integer)*3600 +cast(substr(CR_RETURNED_DATE_SK,4,2) as integer)*60 +cast(substr(CR_RETURNED_DATE_SK,7,2) as integer)) = t_time)
left outer join item on (to_char(CR_ITEM_SK) = i_item_id)
left outer join customer c1 on (to_char(CR_RETURNING_CUSTOMER_SK) = c1.c_customer_id)
left outer join customer c2 on (to_char(CR_REFUNDED_CUSTOMER_SK) = c2.c_customer_id)
left outer join reason on (to_char(CR_REASON_SK) = r_reason_id)
where i_rec_end_date is NULL;
select count(*) from catalog_returns;
select count(*) from crv;