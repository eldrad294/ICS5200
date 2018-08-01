# Using the TPC-DS tools
## Data generation
Data generation is done via dsdgen. See dsdgen --help for all options. If you do not run dsdgen from the tools/ directory then you will need to use the option -DISTRIBUTIONS /.../tpcds-kit/tools/tpcds.idx.

## Query generation
Query generation is done via dsqgen. See dsqgen --help for all options.

The following command can be used to generate all 99 queries in numerical order (-QUALIFY) for the 10TB scale factor (-SCALE) using the Netezza dialect template (-DIALECT) with the output going to /tmp/query_0.sql (-OUTPUT_DIR).

dsqgen \
-DIRECTORY ../query_templates \
-INPUT ../query_templates/templates.lst \
-VERBOSE Y \
-QUALIFY Y \
-SCALE 10000 \
-DIALECT netezza \
-OUTPUT_DIR /tmp