DECLARE
   max_sk NUMBER;
   i_count number;
BEGIN
   select count(*)
   into i_count
   from all_sequences
   where sequence_name = upper('item_seq');
   if i_count > 0 then
   execute immediate 'drop sequence item_seq';
   end if;
   SELECT max(i_item_sk)+1 INTO max_sk FROM item;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE item_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;