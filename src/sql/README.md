This directory houses all relevant SQL scripts utilized in the project, distributed as follows:
* Installation - Setup scripts, required to run only once at inception of projection installation
* Rollback - Teardown scripts, required for un-installation purposes
* Runtime - Executable/repetitive scripts utilized during life cycle of this project
* Utility - Scripts which are useful, and have therefore been kept handy in case of future use

Ensure that the relevant instance has ample allocated processes and session before data migration job commences, as
follows. Since Spark makes use of a number of parallel executors, each with their own database connection, this can
prove taxing on the instance:
* https://knowledgebase.progress.com/articles/Article/P164971
* http://www.dba-oracle.com/t_ora_12516-tns_ould_not_find_available_handler.htm