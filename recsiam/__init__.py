
from . import utils
# from . import models
import logging
from pathlib import Path

logfile = str(Path(__file__).parent / "debug.log")

logger = logging.getLogger('recsiam')
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
