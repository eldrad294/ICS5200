This repository is dedicated for housing of Gabriel Sammut's ICS5200 Master's Dissertation

# Intro
This M.Sc research dissertation project attempts an untried approach for relational database optimizer statistic upkeep through usage of machine learning methods. Primary focus is the attempt at 
maximizing Optimizer Statistics accuracy, by ensuring that all relavent statistics are kept up to date, respective of database schema and data. This ensures that any capable SQL engine attempting
to generate efficient Explain Plans can do so upon mantained, updated Optimizer Statistics.

# Virtual Environment
Project dependencies are logged under a Python3 virtual environment. The enviroment can be enabled/disabled using the following instructions:
## Enable Virtual Environment
* cd ~/ICS5200/venv/bin
* source activate
## Disable Virtual Environment
* cd ~/ICS5200/venv/bin
* deactivate 

# Project Distribution
This repository is distributed as follows
## bash
  * benchmark - Bash scripts reserved to initiate TPC benchmark
  * setup - Bash scripts reserved to initiate TPC installation
  * spark - Bash scripts reserved for Spark startup/cleanup
## data
  * TPC-DS
  * TPC-E
## docs
  * Project related documentation
## log
  * msg_log_yyyymmdd
## src
  * data (Contains TPC Related tools + setup)
  * framework (Contains core framework logic referenced project wide)
  * main (Contains main executable scripts driving project flow)
  * prototypes (Contains rough sketches / primitive / initial idea workflows)
  * reports (Contains visualization scripts based atop TPC schemas)
  * sql (Contains repository of all SQL referenced in this project)
  * utils (Contains project independent scripts, which can be used as standalone logic if needs be)
## venv
  * python3 virtual environment
## visuals
  * Directory reserved for housing of html generated reports/graphs


