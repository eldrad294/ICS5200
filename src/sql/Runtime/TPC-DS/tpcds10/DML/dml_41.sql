update web_returns set WR_RETURNED_DATE_SK = null where WR_RETURNED_DATE_SK like '%-47%';
drop table wrv;
create table wrv tablespace tpcds_benchmark as
select d_date_sk wr_return_date_sk
      ,t_time_sk wr_return_time_sk
      ,i_item_sk wr_item_sk
      ,c1.c_customer_sk wr_refunded_customer_sk
      ,c1.c_current_cdemo_sk wr_refunded_cdemo_sk
      ,c1.c_current_hdemo_sk wr_refunded_hdemo_sk
      ,c1.c_current_addr_sk wr_refunded_addr_sk
      ,c2.c_customer_sk wr_returning_customer_sk
      ,c2.c_current_cdemo_sk wr_returning_cdemo_sk
      ,c2.c_current_hdemo_sk wr_returning_hdemo_sk
      ,c2.c_current_addr_sk wr_returing_addr_sk
      ,WR_WEB_PAGE_SK wr_web_page_sk
      ,WR_REASON_SK wr_reason_sk
      ,WR_ORDER_NUMBER wr_order_number
      ,WR_RETURN_QUANTITY wr_return_quantity
      ,WR_RETURN_AMT wr_return_amt
      ,WR_RETURN_TAX wr_return_tax
      ,WR_RETURN_AMT_INC_TAX + WR_RETURN_TAX as wr_return_amt_inc_tax
      ,WR_FEE wr_fee
      ,WR_RETURN_SHIP_COST wr_return_ship_cost
      ,WR_REFUNDED_CASH wr_refunded_cash
      ,WR_REVERSED_CHARGE wr_reversed_charde
      ,WR_ACCOUNT_CREDIT wr_account_credit
      ,WR_RETURN_AMT+WR_RETURN_TAX+WR_FEE
       -WR_REFUNDED_CASH-WR_REVERSED_CHARGE-WR_ACCOUNT_CREDIT wr_net_loss
from web_returns left outer join date_dim on (to_char(WR_RETURNED_DATE_SK) = d_date)
                   left outer join time_dim on (( cast(substr(WR_RETURNED_TIME_SK,1,2) as integer)*3600
                                                 +cast(substr(WR_RETURNED_TIME_SK,4,2) as integer)*60
                                                 +cast(substr(WR_RETURNED_TIME_SK,7,2) as integer)) = t_time)
                   left outer join item on (to_char(WR_ITEM_SK) = i_item_id)
                   left outer join customer c1 on (to_char(WR_REFUNDED_CUSTOMER_SK) = c1.c_customer_id)
                   left outer join customer c2 on (to_char(WR_REFUNDED_CUSTOMER_SK) = c2.c_customer_id)
                   left outer join reason on (to_char(WR_REASON_SK) = r_reason_id)
                   left outer join web_page on (to_char(WR_WEB_PAGE_SK) = WP_WEB_PAGE_id)
where i_rec_end_date is NULL
  and wp_rec_end_date is NULL;
select count(*) from wrv where wr_item_sk is null;
select count(*) from web_returns;
select count(*) from wrv;