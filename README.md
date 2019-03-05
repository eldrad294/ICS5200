This repository is dedicated for housing of Gabriel Sammut's ICS5200 Master's Dissertation

# Intro & Description
This M.Sc research dissertation project attempts an untried approach for relational database optimizer statistic upkeep through usage of machine learning methods. Optimizer statistics are an important component of any Cost-Based Optimizer component in a relational database. They serve to supply the optimizer with relavant information so as to produce efficient and optimized access plans. Access plans are a 'map' produced for each and every SQL query executed against the database, dictating the means as to how data is retrieved from the underlying data.

This dissertation focusses at attempts made to maximize the Optimizer Statistics generation process. This is done by ensuring that all relavent statistics are kept up to date, by influencing it's decision making process from respective  database schema and data. We use prior database workloads to better influence the optimizer scheduling process, as well as improving the decision making process required to choose which statistics to gather. Our work is divided into the following components:

* WHEN TO GENERATE STATS? - By learning from prior database activity, we learn to anticipate upcoming peaks/dips of database activity, and therefore be able to schedule the optimizer statistics job during dips of low database activity.
* WHICH QUERIES TO LEARN FROM? - To better and more accurately match the optimizer's need for constant statistical refreshes, we use prior SQL activity to influence our decision making process. Using the notion of a 'query map', we use prior access plans as a baseline, with which we match upcoming optimizer plan predictions to deduce whether plans will change. In the case of a plan change, we pinpoint which access plan node differs, and generate statistics accordingly in an effort to correct statistical inconsistencies and optimizer inefficiencies. 
* WHAT MAKES A WORKLOAD EXPENSIVE? - Since our techniques are influenced from prior database activity, we identify that a workload is composed of multiple factors, cosisting of different type of query consumers. We assume, that not all components of a workload are expensive, and there make effort to identify what elements of a workload are most detrimental. We use outlier detection based from prior access plans, to identify which queries are most expensive amongst others. Therefore, we restrict our focus to these expensive outliers, since they are the type of executuons which are most reliant to have optimized plans. Furthermore, we implement a level of query anticipation, wherein we attempt to predict the sequence of these expensive queries, so as to better influence the optimizer statistical process.

# OS & Environment
Project dependencies are logged under a Python3 virtual environment. Scripts denoted below are meant to function on a unix based system, with the sole exception of the 'notebook' directory which is tested on Windows. The enviroment can be enabled/disabled using the following instructions:
## Enable Virtual Environment
* cd ~/ICS5200/venv/bin
* source activate
## Disable Virtual Environment
* cd ~/ICS5200/venv/bin
* deactivate 

# Project Distribution
This repository is distributed as follows
## bash
  * benchmark - Bash scripts reserved to initiate TPCDS benchmark.
  * report - Bash scripts reserved to iniate TPCDS.
  * setup - Bash scripts reserved to initiate TPCDS installation.
  * spark - Bash scripts reserved for Spark startup/cleanup. This logic is deprecated, since use of Spark was dismissed at early stages of this project.
  * scheduler - Bash scripts used to invoke the workload generation tool. Invoking these scripts not only kicks off the workload benchmark, but also iniates a parallel data gathering thread which polls database metrics every 60seconds.
## data
  * TPC-DS - Contains TPC-DS related/provided tools used to generate data.
  * TPC-E- Contains TPC-E related/provided tools used to generate data. This is deprecated, and was discarded since our approach utilized TPC-DS as a test bed.
## docs
  * Contains minor project related documentation.
## notebook
  * benchmark - Contains ipynb notebooks related to benchmarking each of TPC-DS schemas. We do this to gather an understanding of the behaviour of the TPC-DS components, with respect to different data volumes (1/10/100).
  * gridsearch - Contains python scripts which implement our grid search attempts on RandomForest, Feed Foward NN, and LSTM NNs so as to deduce the best hyperparameter combination.
  * prototypes - Contains code with initial / prototypal work which did not make it into the final submission.
  * schedule - Contains our core experiments. Experiments are divided into four subsets, related to feature descriptions and visualizations, time scheduling, top consumer analysis, incoming query predictions, tree modelling and plan comparison.
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


