with customer_total_return as(select sr_customer_sk as ctr_customer_sk,sr_store_sk as ctr_store_sk,sum(SR_FEE) as ctr_total_returnfrom store_returns,date_dimwhere sr_returned_date_sk = d_date_skand d_year =2000group by sr_customer_sk,sr_store_sk)select * from ( select  c_customer_idfrom customer_total_return ctr1,store,customerwhere ctr1.ctr_total_return > (select avg(ctr_total_return)*1.2from customer_total_return ctr2where ctr1.ctr_store_sk = ctr2.ctr_store_sk)and s_store_sk = ctr1.ctr_store_skand s_state = 'TN'and ctr1.ctr_customer_sk = c_customer_skorder by c_customer_id ) where rownum <= 100;