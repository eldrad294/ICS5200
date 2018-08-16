DECLARE
   max_sk NUMBER;
   i_count number;
BEGIN
   --
   select count(*)
   into i_count
   from all_sequences
   where sequence_name = upper('web_page_seq');
   --
   if i_count > 0 then
     execute immediate 'drop sequence web_page_seq';
   end if;
   SELECT max(WP_WEB_PAGE_SK)+1 INTO max_sk FROM web_page;
   EXECUTE IMMEDIATE 'CREATE SEQUENCE web_page_seq INCREMENT BY 1 START WITH '||max_sk||' ORDER';
END;
/
