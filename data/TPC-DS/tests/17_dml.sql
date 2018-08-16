drop SEQUENCE item_seq;
DECLARE
   max_sk NUMBER;
BEGIN
   SELECT max(i_item_sk)+1 INTO max_sk FROM item;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE item_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;
