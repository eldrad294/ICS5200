TIME SERIES PREDICTIONS

To address the challenges behind optimizer statistics scheduling, an automated intelligent decision support system is being proposed. First, there is first the need to identify the need that is established in this 
domain. With modern optimizer statistics maintenance techniques swaying between hard fixed time windows and database admin manual intervention (who is considered to have a high level of awareness concerning 
contextual information in relation to the underlying database activity), a balance of both is missing. The proposed approach is an artificially intelligent decision making module, wherein intelligent decision 
making behind optimizer statistics scheduling can be automated, to overcome the challenges presented in [12, 30, 43]. Furthermore, to overcome the problems of insufficient optimizer statistic provisioning as 
highlighted by Ziauddin et al., the decision making approach will be directly influenced by the underlying database activity, upon which it will base future predictions and optimally planned scheduling attempts. 
With particular reference to prior approaches to workload characterization, a number of machine learning models will be trained. Both training and evaluation will be carried out upon the acquired 
trace data, for each of the TPCDS schemas. In addition to different trained models, different data transformation techniques will also be considered, in our attempts to address the problem as a 
supervised classification problem.

For the supervised approach of the proposed procedure, each data point vector is assigned a total of two labels. It should however be noted that the type and number of labels to be chosen varies between one 
workload trace and another, and that there is no golden rule behind this choice. Such a decision should be influenced first and foremost by the characteristics of the underlying database activity and those 
metrics which are considered most expressive in a workload. The column labels that were opted are denoted below:

*) CPU_TIME_DELTA - Summed delta value of CPU time (ms) used by database cursors for parsing, executing and fetching.
*) IOWAIT_DELTA - Delta value of user I/O wait time.

These labels were chosen due to their frequent usage in prior literature [53, 87, 92], particularly in the query performance prediction domain. Furthermore, the choice behind these metrics is inspired from the CPU 
and IO hungry properties exhibited by the TPC-DS schema [82]. Metrics attributed to cursor performance [11] are particularly representative of a workload’s activity due to encapsulating behaviour on a query level. 
Each of the following labels are considered descriptive towards identifying RDBMS behavior [151], and can therefore be relied upon to reflect the database’s activity at a single point in time. The chosen labels are 
denoted as ‘deltas’. These deltas denote the amount difference within one particular time window, which are in turn summed respectively as a function over all query activity within that time window.
For each of the three pre-established TPC-DS trace datasets, input features were time shifted by a number of time staggered steps, with respect to output labels. This allows for a lag dynamic which shifts data 
vectors in relation to past, or future labels. The amount of denoted lag shift carried out on the data was extracted from autocorrelation plots. Fig. 3 denotes the amount of correlation (y-axis) as data is shifted 
through different lag amounts (x-axis).


Before exposing the dataset to a number of machine learning models, the dataset was exposed to a number of preprocessing procedures. Each supervised experiment implemented a strict preprocessing pipelined approach 
before training and evaluation of machine learning models. In their body of work, the authors compose a list of preprocessing ‘checks’ as a guide to some of the most recommended preprocessing stages of any machine 
learning experiments. Their recommendations serve to partially eliminate, reduce and transform the number of presented features, which is notably high from the generated trace dataset. This preprocessing phase not 
only enhances the quality of the trained models, but also serves to aid machine learning attempts so as to faster converge. The preprocessing pipelined stages and sequence can be found below.

*) Datasets ‘DBA_HIST_SYSMETRIC_SUMMARY’ and ‘DBA_HIST_SYSSTAT’ are pivoted on their ‘METRIC_NAME’ columns.
*) Replace empty metric values which are void as a result of data collection anomalies or  otherwise, with ‘0’.
*) Data is aggregated and summed on ‘SNAP_ID’ for all datasets. Furthermore, ascending ordering is maintained on the ‘SNAP_ID’ data column so as to retain the natural order of the trace execution.
*) Dataset ‘DBA_HIST_SQLSTAT’, ‘DBA_HIST_SYSMETRIC_SUMMARY’ and ‘DBA_HIST_SYSSTAT’ are merged together into a single data matrix. The join operation is carried out on ‘SNAP_ID’, which is universally present in all 
three datasets. Dataset ‘DBA_HIST_SQL_PLAN’ is not considered for this experiment subset [151].
*) Flatline columns (those exhibiting a natural standard deviation of 0) are removed. Another selection of manually chosen columns (considered non-attributing to the research domain) are also removed.
*) Rows containing values exceeding the interquartile maximum for any particular column are considered outliers. Due to the temporal nature of the dataset, these outliers are not removed, but instead retained and 
transformed to equate a value equal to the interquartile maximum.
*) The dataset is scaled down so as to be commensurate to other features within the dataset, as proposed by Guyon and Elisseeff [100]. Values are scaled between a range of 0 and 1, so as to better help upcoming 
machine learning models to generalize to training data.
*) The dataset is split into respective features and labels. Features will be used to collectively train upcoming supervised machine learning models, in an effort to predict targeted labels. The correctness of each 
trained model will be gauged according to their capability in predicting these labels. 
*) Data is subjected to a lag shifting procedure, as denoted in Table 5. From experiment implemented autocorrelation plots, the most suitable amount of lag is obtained, allowing data to be shifted across ‘n’ time 
shifts (respective of each TPC-DS schema), both in the past (t-n) and future (t+n).
*) Recursive feature elimination is carried out on the lag shifted features. A wrapper feature selection process is implemented through Random Forest feature importance ranking, and three quarters of the entire 
features are removed. The decision behind how many number of features to eliminate was chosen loosely with no definite reason, as it was argued that these features will be further reduced in upcoming preprocessing 
stages.
*) Even with feature elimination techniques in place, we affirm that dataset dimensionality is still high at this point. With a quarter of the original number of features still retained after the feature selection 
process, the dataset is then subjected to a principal component analysis (PCA) transformation, reducing features and overall dimensionality, stopping only when the reduced components go below a hard-defined 0.99 
variance threshold.
*) The remaining feature components are binned to represent a classification problem. Values denoted below the median point are labelled as ‘0’, whilst the rest are labelled as ‘1’. Values of ‘0’and ‘1’ denote 
respective low and high periods of throughput feature activity at a single point in time.
*) The preprocessed result of this data is then used to feed a number of machine learning models. For the established task of supervised classification (which incorporates data of a temporal nature), the following 
techniques are delved deeper within subsections 3.1.1 till 3.1.3. Experiments were implemented in Python, using a mixture of library aided implementations in Scikit Learn particularly for Random Forest 
implementations (Section 3.1.1), as well with Tensorflow backend using Theano for neural network implementations. Each experiment was executed on a single AMD Radeon (TM) 
R9 380 Series GPU, supported by PlaidML GPU speedup.
