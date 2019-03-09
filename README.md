This repository is dedicated for housing of Gabriel Sammut's ICS5200 Master's Dissertation

# Intro & Description
This M.Sc research dissertation project attempts an untried approach for relational database optimizer statistic upkeep through usage of machine learning methods. Optimizer statistics are an important component of any Cost-Based Optimizer component in a relational database. They serve to supply the optimizer with relavant information so as to produce efficient and optimized access plans. Access plans are a 'map' produced for each and every SQL query executed against the database, dictating the means as to how data is retrieved from the underlying data.

This dissertation focusses at attempts made to maximize the Optimizer Statistics generation process. This is done by ensuring that all relavent statistics are kept up to date, by influencing it's decision making process from respective  database schema and data. We use prior database workloads to better influence the optimizer scheduling process, as well as improving the decision making process required to choose which statistics to gather. Our work is divided into the following components:

* WHEN TO GENERATE STATS? - By learning from prior database activity, we learn to anticipate upcoming peaks/dips of database activity, and therefore be able to schedule the optimizer statistics job during dips of low database activity.
* WHICH QUERIES TO LEARN FROM? - To better and more accurately match the optimizer's need for constant statistical refreshes, we use prior SQL activity to influence our decision making process. Using the notion of a 'query map', we use prior access plans as a baseline, with which we match upcoming optimizer plan predictions to deduce whether plans will change. In the case of a plan change, we pinpoint which access plan node differs, and generate statistics accordingly in an effort to correct statistical inconsistencies and optimizer inefficiencies. 
* WHAT MAKES A WORKLOAD EXPENSIVE? - Since our techniques are influenced from prior database activity, we identify that a workload is composed of multiple factors, cosisting of different type of query consumers. We assume, that not all components of a workload are expensive, and there make effort to identify what elements of a workload are most detrimental. We use outlier detection based from prior access plans, to identify which queries are most expensive amongst others. Therefore, we restrict our focus to these expensive outliers, since they are the type of executuons which are most reliant to have optimized plans. Furthermore, we implement a level of query anticipation, wherein we attempt to predict the sequence of these expensive queries, so as to better influence the optimizer statistical process.

# OS & Environment
Project dependencies are logged under a Python3 virtual environment. Scripts denoted below are meant to function on a unix based system (tested on CentOS), with the sole exception of the 'notebook' directory which is run on Windows. The enviroment can be enabled/disabled using the following instructions:
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


# Media Content

Our work is distributed into the following modules:
* Workload Generator Tools (Contains our data generation policies)
* Trace Data (Contains our generated data, TPC-DS 1, 10, 100)
* Experimental Outcomes (Contains our analysis and machine learning attempts on the acquired trace data)

## Workload Generator Tools

This part of the presented project contains our work towards establishing the TPC-DS workload generator tool and integrating it into an Oracle environment. Our work encapsulates the workload generator tools denoted below:

### Setup

This section contains our data and query generation logic, as well as data loading into our Oracle instances. Our setup policies follow this order:

* TPC-DS data files are generated, with 1, 10 and 100 gigabytes respectively. We use dsdgen to generate this data into .dat file format
* Creating the Oracle database instances for 1, 10 and 100. We create our schema strcuture in accordance to TPC-DS specification
* Loading our data (.dat format) into each respective TPC-DS oracle schema. We use SQLLoader to insert our data.
* Creation of our schema indexes, in accordance to TPC-DS specification.
* Generate TPC-DS queries and data maintenance procedures through dsqgen.

### Report

This section contains work related to visualizing our TPC-DS benchmarks. These reports denote the sizing attributed to each TPC-DS schema.

### Benchmark

After loading our data into each respective TPC-DS schema, we benchmark the lack and application of optimizer statistics with respect to each schema. We gauge different metrics pertaining from REP_HIST_SNAPSHOT, by running the query and maintenance jobs before and after running of optimizer statistics. We utilize the following procedure to benchmark the effect of optimizer statistics:

* Drop all schema optimizer statistics on the respective TPC-DS schema.
* Execute all TPC queries generated for each of the TPC-DS schemas. Each query execution plan is extracted and returned/saved to disk inside table REP_EXECUTION_PLANS.
* Execute all TPC DML generated for each of the TPC-DS schemas. Each dml execution plan is extracted and returned/saved to disk inside table REP_EXECUTION_PLANS.
* Repeat Step 2 and 3 for three iterations.
* Generate schema wide optimizer statistics for each of the TPC-DS schemas.
* Execute all TPC Queries generated for each of the TPC-DS schemas. Each query execution plan is extracted and returned/saved to disk inside table REP_EXECUTION_PLANS.
* Execute all TPC DML generated for each of the TPC-DS schemas. Each dml execution plan is extracted and returned/saved to disk inside table REP_EXECUTION_PLANS.
* Repeat Step 6 and 7 for three iterations.

### Scheduler

Our scheduler logic handles execution of the workload benchmark. The workload follows the following procedure (as specified by TPC-DS), over an indefinite period:

* Gather Optimizer Statistics (Collecting schema wide using a high parallel degree)
* Power Test (Execute TPC-DS queries in serial order)
* Throughput Test 1 (Execute TPC-DS queries in parallel fashion across 20 streams)
* Data Maintenance Test 1 (Execute TPC-DS maintenance tasks in serial fashion)
* Throughput Test 2 (Execute TPC-DS queries in parallel fashion across 20 streams)
* Data Maintenance Test 2 (Execute TPC-DS maintenance tasks in serial fashion)
* Repeat from step 1 (We manually terminate the flow 14 days into our workload generation).

## Trace Data

Our trace data consists of the following components, for each of the three TPC-DS schemas (1, 10, 100):

* msg_log_tpcds(X)scheduler - Messsage logs acquired during generation of the trace data. X denotes tpc-ds sizing (1, 10, 100).
* rep_hist_snapshot
* rep_hist_sysmetric_summary
* rep_hist_sysstat
* rep_vsql_plan

## Experimental Outcome

Our experiments are denoted into four sections, for each of the presented TPc-DS schemas (1, 10, 100):

* Time Series Analysis - Contains work in relation to workload characterization and prediction of upcoming database activity.
* Query Sequence Analysis - Contains work in relation to top consumer analysis, and anticipating upcoming query executions.
* Feature Selection - Contains work towards describing and visualizing the collected datasets. Also contains work in relation towards feature selection.
* Access Plan Recommendation - Contains work in relation to access plan to tree representations, and plan comparison.

# Installation Instructions

The proposed artifact is composed of multiple components. For the workload generation component, we use a fresh CentOS7 installation, and use DBCA \cite{RichBert2018Oracle12.2} to install a fresh Oracle 12c database instance. This is followed by installation of a Python 3.6 environment. Our utilities can be executed from the following directory:

ICS5200/bash/
* benchmark - Bash scripts geared towards bench marking the workload's components with and without presence of optimizer statistics.
* report - Bash scripts geared towards producing reports after each of the TPC-DS schemas have been populated.
* scheduler - Bash scripts geared towards handling of the trace generation.
* setup - Bash scripts for TPC-DS 1, 10, 100, used to create workload data and load it into an Oracle instance.
* spark - Deprecated. At the beginning we briefly considered Spark for loading of the generated data into our Oracle schemas. This was eventually replaced with SQL Loader \cite{GennickJonathan2001OracleGuide}.

For the experimental section of our work, wherein we apply a number of machine learning techniques on the acquired trace data, we denote the following installation guideline. We carry out these experiments on a windows installation, installed on either Windows 8 or 10 (both have been verified to work). We recommend a fresh Anaconda python environment, updated to at least version 3.6. All python modules mentioned in Table must be installed.

# User Manual

This manual is distributed into two sections. Each section is composed into a number of procedural steps.

## Workload Generator

Before executing these set of steps for workload generator, we remind the user that the following steps are geared towards executing the workload generator so as to generate the RDB trace data, and therefore is not required to trigger the generator. This data is supplied with the final deliverable, for each TPC-DS schema. For the sake of replicability, we suggest the following steps:

* Navigate to ~/ICS5200/src/main
* Open 'config.ini' file with text editor of choice.
* Edit the following config parameters according to your setup:
 - oracle_home: Oracle home installation path.
 - ld_library_path: Oracle library path.
 - instance_name: Oracle instance name.
 - sysuser: Oracle root username.
 - user: Schema username. We recommend using tpcds1, tpcds10, tpcds100 respectively.
 - host: Server host IP address where Oracle instance is setup.
 - service: Oracle instance service.
 - port: Service port number.
 - syspassword: Oracle root password.
 - password: Oracle schema password.
 - data_generated_directory: Path directory where the generated TPC-DS .dat files are generated by dsdgen.
 - sql_generated_directory: Path where the generated TPC-DS SQL workload scripts are generated by dsqgen.
 - data_size: Denotes data size of generated TPC-DS workload. We recommend multiples of 10 as a value.
 - parallel_degree: Number of parallel threads to invoke for data generation tasks.
 - statistic_intervals: Number of seconds for metric extraction workload polling mechanism.
 - stream_total Number of parallel streams invoked during workload generator. As specified by TPC-DS \cite{TransactionProcessingCouncil2018TPCDS}, this should default to 20, but a smaller number is recommended for smaller server architectures. 
* Navigate to '~/ICS5200/bash'
* Execute the functionality the respective functionality.

## Experiments

For the experimental sections of our work, we suggest the following steps:

* Launch Jupyter in '~/ICS5200'
* Navigate to '~/ICS5200/notebook/schedule/TPC-DSX/Schedule/TPC-DSX/.
* Open .ipynb notebook of choice.
* Each notebook contains a configuration cell, which contains paths which point towards the gathered trace data. These will need to be adjusted respective to the underlying machine.
* Run the notebook.