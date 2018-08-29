#!/bin/bash
cd /home/gabriels/ICS5200/venv/bin
source activate
cd ../../
nohup python3 src/main/benchmark.py > log/nohup_output_tpcds_benchmark10.log &