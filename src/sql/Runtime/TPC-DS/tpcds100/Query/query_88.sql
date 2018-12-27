SELECT *
FROM   (SELECT Count(*) h8_30_to_9
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 8
               AND time_dim.t_minute >= 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s1,
       (SELECT Count(*) h9_to_9_30
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 9
               AND time_dim.t_minute < 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s2,
       (SELECT Count(*) h9_30_to_10
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 9
               AND time_dim.t_minute >= 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s3,
       (SELECT Count(*) h10_to_10_30
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 10
               AND time_dim.t_minute < 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s4,
       (SELECT Count(*) h10_30_to_11
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 10
               AND time_dim.t_minute >= 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s5,
       (SELECT Count(*) h11_to_11_30
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 11
               AND time_dim.t_minute < 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s6,
       (SELECT Count(*) h11_30_to_12
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 11
               AND time_dim.t_minute >= 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s7,
       (SELECT Count(*) h12_to_12_30
        FROM   store_sales,
               household_demographics,
               time_dim,
               store
        WHERE  ss_sold_time_sk = time_dim.t_time_sk
               and ss_item_sk between 95700 and 100000
               and ss_ticket_number between 36615 and 40000
               AND ss_hdemo_sk = household_demographics.hd_demo_sk
               AND ss_store_sk = s_store_sk
               AND time_dim.t_hour = 12
               AND time_dim.t_minute < 30
               AND ( ( household_demographics.hd_dep_count = 1
                       AND household_demographics.hd_vehicle_count <= 1 + 2 )
                      OR ( household_demographics.hd_dep_count = -1
                           AND
household_demographics.hd_vehicle_count <=- 1 + 2 )
                      OR ( household_demographics.hd_dep_count = 2
                           AND
household_demographics.hd_vehicle_count <= 2 + 2 ) )
               AND store.s_store_name = 'ese'
               and rownum <= 10000) s8;