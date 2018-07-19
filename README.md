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
- src
  - datageneration
  - main
  - prototypes
  - sql
  - utils
- data
  - TPC-DS
  - TPC-E
