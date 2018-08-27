#
# Import Modules
import os
class FlashbackControl:
    #
    @staticmethod
    def create_restore_point(logger, ev_loader, restore_point_name):
        """
        This method needs to be re-written to support table flashback mode
        :param logger:
        :param ev_loader:
        :param restore_point_name:
        :return:
        """
        cmd = "exit | sqlplus " + ev_loader.var_get('user') + "/" + ev_loader.var_get('password') + "@" \
              + ev_loader.var_get('instance_name') + " @" + ev_loader.var_get('src_dir') + \
              "/sql/Utility/create_restore_point.sql " + restore_point_name
        logger.log(cmd)
        output = os.system(cmd)
        if output != 0:
            logger.log("Exception raised during generation of TPC files..Terminating process!")
            raise Exception("Exception raised during generation of TPC files..Terminating process!")
        logger.log('Created restore point [' + restore_point_name + ']')
