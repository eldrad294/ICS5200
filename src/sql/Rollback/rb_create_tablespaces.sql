/*
Drop Tablespaces + Datafiles

* Tablespace | Datafiles can be checked from:
select * from dba_tablespaces;
select * from DBA_DATA_FILES;
*/
DROP TABLESPACE tpcds1   INCLUDING CONTENTS;
DROP TABLESPACE tpcds10  INCLUDING CONTENTS;
DROP TABLESPACE tpcds50  INCLUDING CONTENTS;
DROP TABLESPACE tpcds100 INCLUDING CONTENTS;
