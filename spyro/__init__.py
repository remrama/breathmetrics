import logging
from .detection import *
from .preprocessing import *
from .respiration import *
from .utils import *

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%b-%d %H:%M:%S")

__version__ = "0.0.0"
