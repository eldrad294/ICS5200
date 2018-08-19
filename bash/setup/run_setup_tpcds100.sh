#!/usr/bin/env bash
cd /home/gabriels/ICS5200/venv/bin
source activate
cd ../../
export ORACLE_HOME=/oracle/product/11.2.0/dbhome_1
export LD_LIBRARY_PATH=/oracle/product/11.2.0/dbhome_1/lib
nohup python3 src/main/setup.py > log/nohup_output_tpcds_setup100.log &