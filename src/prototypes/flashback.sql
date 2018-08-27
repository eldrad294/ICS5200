--
-- Creates a restore point
CREATE RESTORE POINT before_upgrade;
CREATE RESTORE POINT before_upgrade GUARANTEE FLASHBACK DATABASE;
--
-- Retrieves all restore points
SELECT NAME, SCN, TIME, DATABASE_INCARNATION#,
GUARANTEE_FLASHBACK_DATABASE,STORAGE_SIZE
FROM V$RESTORE_POINT;
--
-- Removes restore points
DROP RESTORE POINT before_upgrade;
--
select *
from CATALOG_PAGE;
--
delete
from CATALOG_PAGE
where rownum <= 718;
commit;
--
flashback table catalog_page to timestamp to_date('27-AUG-2018 17:58:00','DD-MON-YYYY HH24:MI:SS');