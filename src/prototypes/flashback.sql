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