insert into web_returns (select * from wrv where WR_ITEM_SK is not null);
rollback;