#!/bin/bash
export ORACLE_HOME=/oracle/product/12.1.0/dbhome_1
export ORACLE_SID=gabsam
cd /home/gabriels/ICS5200/venv/bin
source activate
cd ../../
nohup python3 src/main/benchmark.py > log/nohup_output_tpcds_benchmark100.log &