import os
from datetime import datetime
import logging
log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(), 'logs',log_file)
os.makedirs(logs_path,exist_ok=True)
log_file_path=os.path.join(logs_path,log_file)
logging.getLogger("azure").setLevel(logging.WARNING)  # or ERROR

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s ] %(lineno)d - %(levelname)s - %(message)s",
    level=logging.DEBUG)

if __name__=="__main__":
    logging.info("logging started")