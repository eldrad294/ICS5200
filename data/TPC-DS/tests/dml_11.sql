DECLARE
   max_sk NUMBER;
   i_count number;
BEGIN
   select count(*)
   into i_count
   from all_sequences
   where sequence_name = upper('web_site_seq');
   if i_count > 0 then
     execute immediate 'drop sequence web_site_seq';
   end if;
   SELECT max(web_site_sk)+1 INTO max_sk FROM web_site;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE web_site_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;