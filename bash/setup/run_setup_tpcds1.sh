#!/bin/bash
cd /home/gabriels/ICS5200/venv/bin
source activate
cd ../../
nohup python3 src/main/setup.py > log/nohup_output_tpcds_setup1.log &