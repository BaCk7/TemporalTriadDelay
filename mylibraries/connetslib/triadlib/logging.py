
import logging

def setup_log_file(LOG_FILE):

    logging.basicConfig(filename=LOG_FILE,
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%d-%b-%y %H:%M:%S',
                                level=logging.DEBUG)

    
def printlog(*params):
    comment = " ".join([str(item) for item in params])
    # now(comment)
    logging.info(comment)
#     return comment


# LOG_FILE = "snapgenerator_log.txt"
# setup_log_file(LOG_FILE)
# printlog("Start", LOG_FILE)