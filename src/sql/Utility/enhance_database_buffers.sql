--http://databasequest.blogspot.com/2016/12/resize-or-increase-redo-log-size.html
SELECT
    a.GROUP#,
    a.THREAD#,
    a.SEQUENCE#,
    a.ARCHIVED,
    a.STATUS,
    b.MEMBER    AS REDOLOG_FILE_NAME,
    (a.BYTES/1024/1024) AS SIZE_MB
FROM v$log a
JOIN v$logfile b ON a.Group#=b.Group#
ORDER BY a.GROUP# ASC;
--
-- Create larger redo log files
alter database add logfile group 4  '/mnt/raid5/oradata/gabsam/redo04.log' size 2g;
alter database add logfile group 5  '/mnt/raid5/oradata/gabsam/redo05.log' size 2g;
alter database add logfile group 6  '/mnt/raid5/oradata/gabsam/redo06.log' size 2g;
--
--
alter system switch logfile;
alter system checkpoint;
--
-- Drop previous smaller files
alter database drop logfile group 1;
alter database drop logfile group 2;
alter database drop logfile group 3;
alter database drop logfile group 4;
alter database drop logfile group 5;
alter database drop logfile group 6;
--
alter database add logfile group 1  '/mnt/raid5/oradata/gabsam/redo01.log' size 2g reuse;
alter database add logfile group 2  '/mnt/raid5/oradata/gabsam/redo02.log' size 2g reuse;
alter database add logfile group 3  '/mnt/raid5/oradata/gabsam/redo03.log' size 2g reuse;
--
-- Enhance Log Buffer Size
-- https://community.toadworld.com/platforms/oracle/w/wiki/1328.redo-log-buffer-modifying-the-size
select name, value from v$parameter where name = 'log_buffer';
--alter system set log_buffer=1073741824 scope=spfile;
alter system set log_buffer=104857600 scope=spfile;
