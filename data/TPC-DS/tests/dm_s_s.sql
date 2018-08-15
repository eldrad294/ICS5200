drop SEQUENCE store_seq;
DECLARE
   max_sk NUMBER;
BEGIN
   SELECT max(s_store_sk)+1 INTO max_sk FROM store;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE store_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;
