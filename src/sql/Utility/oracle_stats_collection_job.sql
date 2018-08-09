/*
This script enables/disables oracle stats collection job
https://oracledbastories.wordpress.com/2012/08/31/disabling-auto-optimizer-stats-in-oracle-11g/ 

*) SELECT CLIENT_NAME,STATUS FROM DBA_AUTOTASK_CLIENT;

*/
BEGIN
  DBMS_AUTO_TASK_ADMIN.disable(
    client_name => 'auto optimizer stats collection',
    operation   => NULL,
    window_name => NULL);
EXCEPTION
  WHEN OTHERS THEN
    DBMS_OUTPUT.PUT_LINE(SQLERRM);
END;
/

