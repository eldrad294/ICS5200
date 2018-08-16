DECLARE
   max_sk NUMBER;
   i_count number;
BEGIN
	select count(*)
	into i_count
	from all_sequences
	where sequence_name = upper('store_seq');
	--
	if i_count >0 then
	  execute immediate 'drop sequence store_seq';
	end if;
   SELECT max(s_store_sk)+1 INTO max_sk FROM store;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE store_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;