This directory contains 3 seperate notebooks. Each notebook is geared towards the interpretation of query plans as cyclic tree representations. These tree 'models' allow for two plans to be compared to each 
other. Changes in the tree structure can be detected in such manner. A measure of how far apart two tree are is denoted, by comparing tree node expense in terms of CPU cost, IO cost, consumed Bytes, estimated
Cardinality, estimated Time to compute, optimizer attributed costing, and Temp space usage.

*) Access Plan Comparison (Variant to Trace) - Compares variant queries to the original query representation from the trace dataset. Note that since variants are based off the trace dataset, and exhibit slight 
changes in their SQL structure, their respective SQL_ID is expected to change. This makes it difficult to link back to their original representation in the trace dataset. To try and establish their original SQLsyntax
form in the trace dataset, a Levensthein string measure is used on each query plan, to try and match to the closest plan. This approach does not garauntee that query patterns are matched to their original trace
query.

*) Access Plan Comparison (Variant to Variant) - Compares variant queries to other variant datasets. This approach garauntees that the same query templates are matched to their respective counterparts, with whatever
present tree modifications.

*) Access Plan Outlier Recommendation - This module is dedicated to profiling of expensive SQL queries located within the trace dataset. SQL access plans are categorized as being either inliers (considered to operate
normally in terms of incurred expense in relation to the trace dataset), or as outliers (considered to be the most sophisticated and complex to compute within the trace dataset).

By referring to prior generated access plans and relating them to past utilized access plan predicates and overall plan structures, we can establish a baseline of past, stable performance profiles per SQL statement. 
Using this baseline to compare future generated access plans, we can analyze and contrast plan differences wherever they might occur. This approach not only allows for detection of differently structured access plans 
as they might occur during day to day activity, but it also allows for detailed, precise recommendations as to where an optimizer plan might differ from another. By looking at which access predicates makes a plan 
different from prior established ones, recommendations can be given as to which type of statistics and which schema entities should be focused during the statistical maintenance phase. Of course, it must be stated 
that comparisons carried out to prior access plans is done so under the assumption that prior access plans are stable and optimum. Therefore, the proposed method does not cater for eventualities when optimizer 
access plans were not optimal from inception. Instead it focuses on recommendations and comparisons whenever a plan is detected to have changed from a past optimal state.

To achieve a method of suitable comparison, access plans are modelled as traversable, acyclic, binary trees, wherein each node denotes a particular row like step which composes an access plan predicate [136-139]. 
The proposed approach then iterates in a post order fashion on the access tree plan, starting from child access predicates first as the tree is traversed bottom up. Each node is compared to respective and prior 
established access tree plans for the same SQL statement, and any differences are denoted as potential optimizer statistic degradation. Such a technique is inspired from [140]. This technique also caters for plan 
inconsistencies as discussed by [141], wherein plan changes caused as a result of tree node insertion, deletion, and updates, are still captured. Furthermore, the proposed technique does not only check for tree 
plan inconsistencies, but it also incorporates a distance based outlier detection mechanism, wherein node vectors (composed of optimizer attributes) are compared, and flagged as inconsistent in case of large 
discrepancies. Therefore even if access plans are compared and each tree node is considered the same as prior versions of the same query plan, yet exhibits different costings attributed to specific tree nodes, 
inconsistencies are also flagged, provided that the calculated costing exceeds a particular pre-configured threshold. The same optimizer attributes that are used in Table 8 will be applied for the costing 
measurement and comparison of two access plans. Cross comparison and evaluation with outlier queries mentioned in section 2.2.3 will be carried out. Plans pertaining in the original trace workload are compared 
to respective query variants, so as to ascertain that differences present between two access plans are highlighted. 
