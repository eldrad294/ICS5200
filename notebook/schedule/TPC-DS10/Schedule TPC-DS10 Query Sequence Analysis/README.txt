TOP CONSUMER ANALYSIS

Establishing which SQL activity is likely to feature in upcoming, future, time slots is however not sufficient alone. It would be counterproductive and resource inefficient to consider and analyze every SQL pattern 
and activity which is predicted to occur in upcoming time slots, due to the sheer magnitude of certain workloads. Therefore there is a need to identify which SQL activity is most detrimental to the database’s system
wide performance. The proposed approach should recognize which SQL activity is most likely to be expensive in terms of RDBMS resource activity, and which SQL syntax is considered most complex in terms of optimizer 
complexity. Such complex SQL patterns usually tend to benefit the most from carefully maintained optimizer statistics, and are therefore good candidates for pre-emptive monitoring in case of access plan deficiencies. By restricting this analysis to only the most complex of SQL activity within a workload trace, the amount of required predictions can be reduced to a feasible amount. Furthermore, predictions can be established with a higher degree of confidence due to focusing primarily on complex and workload intensive access plans. This subset of complex queries and respectively generated access plans can be considered as workload outliers. They can be distinguished from other queries through the amount of additional incurred database resources, increased plan access predicates and expensive optimizer attributed costing. 
With a number of pre-established techniques in prior literature dedicated towards identification of outliers within a data space, the proposed approach will implement a number of outlier detection techniques to 
detect which SQL activity is considered as a likely candidate for optimizer statistic recommendation. Particular inspiration from [119, 137] is used, for which SQL complexity is evaluated through respective 
attributed access plans. Access plans, which are multi-row by nature [18], are aggregated as a total summed vector of its overall costing. This vector representation allows for queries to be represented and 
compared in a Euclidean distance based approach. Each vector is composed of the following attributes, present for every SQL query. The following attributes are used:

*) COST - Cost attributed to query by the optimizer
*) TEMP_SPACE - Temporary space usage by the SQL execution.
*) IO_COST - I/O cost attributed to executing a query. This metric usually equates to the number of disk hops required by the SQL execution.
*) BYTES - Number of estimated bytes produced by the SQL execution
*) CARDINALITY - Number of estimated rows produced by SQL execution
*) TIME - Estimated elapsed time in seconds for SQL execution completion.

A number of outlier detection mechanisms are opted for, and contrasted together. Initially, a particular usage of a top K-NN implementation is used to rank the highest vectors within the trace workload. This 
approach however does not provide outlier isolation, instead merely ranking access plan vectors by order of magnitude. It can be argued that a threshold value can be opted for, in doing so successfully isolating 
the top N outliers. Note that this technique was implemented in an unsupervised manner, and was not considered to be comparable to later experiments carried in this domain. Instead, the attempts here act as a 
rough heuristic and initial ground work at tackling the problem of query plan clustering and outlier detection.

Further to the above mechanism, an unsupervised learning attempt is carried out.  Through implementation of clustering based mechanisms, data is categorized according to natural underlying groupings. Application 
of the K-Means algorithm is opted for, in an effort to allow vectorized access plans to be clustered according to randomly initialized centroids [71, 137]. Specifically, the K-Means++ initialization method is 
used for placement of initial centroids [156]. For evaluation of inlier and outlier vector counts, vectors are considered outliers if any of its feature values exceeds a threshold of three standard deviations 
greater than expected. Actual estimated counts for inliers vs those predicted through the K-Means clustering mechanism are compared.  Models exhibiting the least amount of outlier count difference from the actual 
counts are considered to be the most effective at clustering outliers. Different ‘K’ configurations are attempted by manually fitting a number of models with different K parameters. Fixed random seeding is used 
to account for randomness of initial centroid positioning. Finally, the Isolation forest bagging machine learning technique is opted for, as described in.  Similarly, a number of different isolation forest 
executions is carried out, accounting for different contamination parameter values [116]. The same outlier ‘labeling’ mechanism used to identify outliers in the K-Means method is applied for Isolation Forest 
evaluation, denoting any vectors whose any singular value exceeding three standard deviations to be considered as outliers. In this context, accuracy is defined as the capability to correctly identify an SQL 
statement pertaining to the workload as either standard (inlier) or expensive (outlier) in overall cost to compute. F1 Scoring, and Receiver Operating Characteristic curves [157] are also used to compare the 
effectiveness of each method. A binary F1 scoring weighting is used, wherein output labels are categorized into one of two outputs: Inlier or Outlier. Each of the experiments (K-NN, K-Means, Isolation Forests), 
and the evaluation methods used to attribute scoring to each trained model, were implemented in Python, using Scikit Learn.

QUERY SEQUENCE PREDICTION

This area of research highlights a particular importance upon optimizer statistics decision making (which is reliant upon past SQL generated access plans), making it useful to anticipate future SQL activity. 
This ability to predict incoming SQL execution, allows the proposed approach to utilize respective SQL access plans, as a map to further guide which statistics are considered most important. Furthermore, the 
optimizer generated plan also shows access predicates for a particular query, which allows the proposed approach to be able to deduce which schema objects are relevant for statistical upkeep. Therefore the 
objective for this sub category of query sequence analysis is to be able to predict upcoming SQL executions, based on prior SQL activity.

To achieve this, the captured trace data pertaining to the TPC-DS workload is split into a number of time window slices, which by default equate to 60 seconds. Each time slice is expected to have a number of 
SQL activity and executions. SQL activity is denoted by the respective syntax SQL_ID, which is unique per SQL query. As the benchmark progresses, certain SQL executions are expected to feature again in future 
time windows. Therefore, based on past occurrence of prior SQL executions, the proposed approach proposes a classification, supervised, machine learning approach. Each SQL is by default denoted by a unique SQL 
identifier (SQL_ID), which is respectively transformed into an encoded representation native to the machine learning model. The aforementioned supervised model produces predictions in the form of encoded labels
(SQL_IDs), for a particular time slice. For neural-net based models, a one hot encoding scheme is used on the dataset. Lag shifting similar to that proposed in Table 5 is carried out, allowing predictions to be 
made in more than a single time step at a time. Only SQL_IDs considered to be outliers (section 3.2.1) are retained for the purposes of this supervised experiment.
