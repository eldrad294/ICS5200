select resource_name, current_utilization, max_utilization, limit_value
    from v$resource_limit
    where resource_name in ('sessions', 'processes');
alter system set processes = 600 scope=spfile;  -- Requires SQL Plus execution + DB Bounce