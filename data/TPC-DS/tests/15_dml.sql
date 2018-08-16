DECLARE
   i_count number;
BEGIN
   --
   select count(*)
   into i_count
   from all_sequences
   where sequence_name = upper('callcenter_seq');
   --
   if i_count > 0 then
   --
   		execute immediate 'drop sequence callcenter_seq';
   end if;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE callcenter_seq INCREMENT BY 1 START WITH 1 ORDER';
END;