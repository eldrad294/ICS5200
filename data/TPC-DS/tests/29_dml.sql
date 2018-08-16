DECLARE
   max_sk NUMBER;
BEGIN
  FOR ca_rec IN (SELECT CUST_CUSTOMER_ID
                       ,CUST_STREET_NUMBER
                       ,STREET
                       ,CUST_STREET_TYPE
                       ,CUST_SUITE_NUMBER
                       ,CUST_CITY
                       ,CUST_COUNTY
                       ,CUST_STATE
                       ,CUST_ZIP
                       ,CUST_COUNTRY
                 from cadrv) LOOP
    update customer_address set
 CA_STREET_NUMBER=ca_rec.CUST_STREET_NUMBER
,CA_STREET_NAME=substr(ca_rec.CUST_STREET_NUMBER,60)
,CA_STREET_TYPE=ca_rec.CUST_STREET_TYPE
,CA_SUITE_NUMBER=ca_rec.CUST_SUITE_NUMBER
,CA_CITY=ca_rec.CUST_CITY
,CA_COUNTY=ca_rec.CUST_COUNTY
,CA_STATE=ca_rec.CUST_STATE
,CA_ZIP=ca_rec.CUST_ZIP
,CA_COUNTRY=ca_rec.CUST_COUNTRY;
  END LOOP;
commit;
END;