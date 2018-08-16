drop SEQUENCE item_seq;
DECLARE
   max_sk NUMBER;
BEGIN
   SELECT '1'
   INTO max_sk
   FROM dual;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE callcenter_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;

