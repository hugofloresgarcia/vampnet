import logging
from pathlib import Path
ROOT = Path(__file__).parent

import numpy as np
from queue import Queue

# make a log file!!
logfile= ROOT / "log.txt"
if logfile.exists():
    logfile.unlink()
logging.basicConfig(filename=logfile, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s | %(levelname)s | %(message)s")


def hsv_to_rgb(h, s, v):
    # from https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
    c = v * s
    h_ = h / 60
    x = c * (1 - abs(h_ % 2 - 1))
    m = v - c

    if h_ < 1:
        r, g, b = c, x, 0
    elif h_ < 2:
        r, g, b = x, c, 0
    elif h_ < 3:
        r, g, b = 0, c, x
    elif h_ < 4:
        r, g, b = 0, x, c
    elif h_ < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return r + m, g + m, b + m


def dbg(*args):
    print(" ".join(map(str, args)))


# we'll want to log on a separate thread 
# so that we can log without blocking the main thread

# make a queue for logging
log_queue = Queue()

# log to a file instead of the console
def log(msg):
    # log_queue.put(msg)
    logging.info(msg)
    pass

def set_debug(debug):
    if debug:
        # print log to console
        logging.getLogger().addHandler(logging.StreamHandler())


def pow2db(x):
    return 10 * np.log10(x + 1e-6)


def db2pow(x):
    return 10 ** (x / 10)
