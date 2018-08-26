declare
  --
  i_count number := 0;
  i_count2 number := 0;
  row_count number;
  idx_count number;
  v_dml varchar2(400);
  v_tpc_type varchar2(8);
  v_report_tablename varchar2(40);
begin
  /*
  Declare config parameters downhere
  */
  --
  v_tpc_type := 'tpcds100';
  --
  -----------------------------------
  -- DO NOT CHANGE BELOW THIS LINE --
  -----------------------------------
  --
  select count(*)
  into i_count
  from dba_tables
  where tablespace_name = upper(v_tpc_type);
  --
  if i_count > 0 then
    begin
	    --
	    v_dml := 'CREATE TABLE REP_TPC_DESCRIBE (tablename varchar2(100),'||
				 	'row_count number,'||
				 	'index_count number) tablespace users';
	    execute immediate v_dml;
	    dbms_output.put_line('Generated table [REP_TPC_DESCRIBE]');
    exception
      when others then
	    dbms_output.put_line('Table [REP_TPC_DESCRIBE] already exists');
    end;
    --
	v_dml := 'select count(*) from REP_TPC_DESCRIBE';
	execute immediate v_dml into i_count;
	--
	if i_count = 0 then
	  --
	  for rec in (select table_name
	              from dba_tables
	 			  where tablespace_name = upper(v_tpc_type))
	  loop
	    --
	    v_dml := 'select /*+PARALLEL(60)*/ count(*) from '||upper(v_tpc_type)||'.'||rec.table_name;
	    execute immediate v_dml into row_count;
	    --
	    --dbms_output.put_line('1');
	    v_dml := 'select count(*) from dba_indexes where table_name = '''|| upper(rec.table_name)||''' and table_owner = '''||upper(v_tpc_type)||'''';
	    execute immediate v_dml into idx_count;
	    --
	    v_dml := 'insert into REP_TPC_DESCRIBE values ('''||upper(rec.table_name)||''','||row_count||','||idx_count||')';
	    execute immediate v_dml;
	  end loop;
	  commit;
	  --
	  dbms_output.put_line('Report Data Generated');
	else
	  --
	  dbms_output.put_line('Data already exists for tpc type ['||v_tpc_type||']');
	end if;
  else
    dbms_output.put_line('Tables for tpc type ['||v_tpc_type||'] does not exist!');
  end if;
exception
  when others then
    dbms_output.put_line(sqlerrm);
end;
/
