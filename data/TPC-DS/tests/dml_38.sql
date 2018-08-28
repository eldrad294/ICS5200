drop table srv;
create table srv tablespace tpcds_benchmark as
select d_date_sk sr_returned_date_sk
      ,t_time_sk sr_return_time_sk
      ,nvl(i_item_sk,0) sr_item_sk
      ,c_customer_sk sr_customer_sk
      ,c_current_cdemo_sk sr_cdemo_sk
      ,c_current_hdemo_sk sr_hdemo_sk
      ,c_current_addr_sk sr_addr_sk
      ,SR_STORE_SK sr_store_sk
      ,SR_REASON_SK sr_reason_sk
      ,SR_TICKET_NUMBER sr_ticket_number
      ,SR_RETURN_QUANTITY sr_return_quantity
      ,SR_RETURN_AMT sr_return_amt
      ,SR_RETURN_TAX sr_return_tax
      ,SR_RETURN_AMT + SR_RETURN_TAX sr_return_amt_inc_tax
      ,SR_FEE sr_fee
      ,SR_RETURN_SHIP_COST sr_return_ship_cost
      ,SR_REFUNDED_CASH sr_refunded_cash
      ,SR_REVERSED_CHARGE sr_reversed_charde
      ,SR_STORE_CREDIT sr_store_credit
      ,SR_RETURN_AMT+SR_RETURN_TAX+SR_FEE
       -SR_REFUNDED_CASH-SR_REVERSED_CHARGE-SR_STORE_CREDIT sr_net_loss
from store_returns left outer join date_dim on (to_char(SR_RETURNED_DATE_SK) = d_date)
                       left outer join time_dim on (( cast(substr(SR_RETURN_TIME_SK,1,2) as integer)*3600
                                                     +cast(substr(SR_RETURN_TIME_SK,4,2) as integer)*60
                                                     +cast(substr(SR_RETURN_TIME_SK,7,2) as integer)) = t_time)
                     left outer join item on (to_char(SR_ITEM_SK) = i_item_id)
                     left outer join customer on (to_char(SR_CUSTOMER_SK) = c_customer_id)
                     left outer join store on (to_char(SR_STORE_SK) = s_store_id)
                     left outer join reason on (to_char(SR_REASON_SK) = r_reason_id)
where i_rec_end_date is NULL
  and s_rec_end_date is NULL;

select count(*) from srv where sr_item_sk is null;
select count(*) from store_returns;
select count(*) from srv;