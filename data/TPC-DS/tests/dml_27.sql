declare
  v_constraint varchar2(60);
begin
  update call_center
  set cc_rec_end_date = sysdate
  where cc_call_center_id in (
    select cc_call_center_id
    from call_center)
  and cc_rec_end_date is NULL;
  select index_name
  into v_constraint
  from user_indexes
  where table_name = 'CALL_CENTER'
  and index_name like 'SYS%';
  execute immediate 'alter table CALL_CENTER drop constraint '||v_constraint;
  insert into call_center(select * from ccv);
  commit;
  execute immediate 'alter table CALL_CENTER add constraint '||v_constraint||' primary key (cc_call_center_sk)';
end;