select          cc_call_center_id Call_Center,        cc_name Call_Center_Name,        cc_manager Manager,        sum(cr_net_loss) Returns_Lossfrom        call_center,        catalog_returns,        date_dim,        customer,        customer_address,        customer_demographics,        household_demographicswhere        cr_call_center_sk       = cc_call_center_skand     cr_returned_date_sk     = d_date_skand     cr_returning_customer_sk= c_customer_skand     cd_demo_sk              = c_current_cdemo_skand     hd_demo_sk              = c_current_hdemo_skand     ca_address_sk           = c_current_addr_skand     d_year                  = 2000 and     d_moy                   = 12and     ( (cd_marital_status       = 'M' and cd_education_status     = 'Unknown')        or(cd_marital_status       = 'W' and cd_education_status     = 'Advanced Degree'))and     hd_buy_potential like '1001-5000%'and     ca_gmt_offset           = -6group by cc_call_center_id,cc_name,cc_manager,cd_marital_status,cd_education_statusorder by sum(cr_net_loss) desc;