drop table cc2_store_returns;
create table cc2_store_returns as (select * from srv);
rollback;
