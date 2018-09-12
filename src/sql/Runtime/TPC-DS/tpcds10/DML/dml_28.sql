DECLARE
   max_sk NUMBER;
BEGIN
  FOR ca_rec IN (SELECT CA_ADDRESS_ID
                       ,CA_STREET_NUMBER AS CUST_STREET_NUMBER
                       ,CA_STREET_NAME AS STREET
                       ,CA_STREET_TYPE AS CUST_STREET_TYPE
                       ,CA_SUITE_NUMBER AS CUST_SUITE_NUMBER
                       ,CA_CITY AS CUST_CITY
                       ,CA_COUNTY AS CUST_COUNTY
                       ,CA_STATE AS CUST_STATE
                       ,CA_ZIP AS CUST_ZIP
                       ,CA_COUNTRY AS CUST_COUNTRY
                 from cadrv) LOOP
    update customer_address set
 CA_STREET_NUMBER=ca_rec.CUST_STREET_NUMBER
,CA_STREET_NAME=substr(ca_rec.STREET,60)
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