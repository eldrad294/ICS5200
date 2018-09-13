#
# Module Imports
from src.framework.db_interface import ConnectionPool
from multiprocessing import Process
import time
#
class Workload:
    #
    __active_thread_count = 0
    #
    @staticmethod
    def execute_transaction(ev_loader, logger, transaction_path, transaction_name):
        p = Process(target=Workload.__execute_and_forget, args=(ev_loader, logger, transaction_path, transaction_name))
        p.start()
    #
    @staticmethod
    def __execute_and_forget(ev_loader, logger, transaction_path, transaction_name):
        """
        This method executes a TPC-DS transaction (query/dml), and left to finish.

        This method is designed to be executed and forgotten. Once executed, this child will no longer be controlled by the
        driver.
        :param ev_loader:
        :param logger:
        :param transaction_path:
        :param transaction_name:
        :return:
        """
        Workload.__active_thread_count += 1
        #
        # Consumes connection from pool
        db_conn = ConnectionPool.claim_from_pool()
        #
        start_time = time.clock()
        db_conn[2].execute_script(user=ev_loader.var_get('user'),
                                  password=ev_loader.var_get('password'),
                                  instance_name=ev_loader.var_get('instance_name'),
                                  filename=transaction_path + "/" + transaction_name,
                                  params=None)
        end_time = time.clock() - start_time
        logger.log('Successfully executed ' + transaction_name + " under " + str(end_time) + " seconds.")
        #
        # Returns connection to pool and makes it available for use
        ConnectionPool.return_to_pool(db_conn)
        #
        Workload.__active_thread_count -= 1

    #
    @staticmethod
    def barrier(ev_loader):
        """
        Halts driver from proceeding any further if active thread count is greater than 'parallel_cap'
        :param ev_loader:
        :return:
        """
        while True:
            if Workload.__active_thread_count < ev_loader.var_get('parallel_cap'):
                break
            else:
                time.sleep(4)