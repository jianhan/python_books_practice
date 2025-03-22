import numpy as np
import math
import time
import numexpr as ne
from loguru import logger

loops = 5000000
a = range(1, loops)


def f(x):
    return 3 * math.log(x) + math.sin(x) * math.log(x)


# using math native function to perform calculation
logger.debug(f"start {loops} loops with native math function")
start = time.time()
r = [f(x) for x in a]
finish = time.time()
elapsed = finish - start
logger.debug(
    f"finish {loops} loops with native math function, elapsed {elapsed:.2f} seconds"
)

# using np array
start = time.time()
logger.debug(f"start {loops} loops with np function")
a = np.arange(1, loops)
r = 3 * np.log(a) + np.sin(a) * np.log(a)
finish = time.time()
elapsed = finish - start
logger.debug(f"finish {loops} loops with np function, elapsed {elapsed:.2f} seconds")


# using numexpr with multithreading
start = time.time()
logger.debug(f"start {loops} loops with numexpr function")
ne.set_num_threads(4)
f = f"3 * log(a) + sin(a) * log(a)"
r = ne.evaluate(f)
finish = time.time()
elapsed = finish - start
logger.debug(
    f"finish {loops} loops with numexpr function, elapsed {elapsed:.2f} seconds"
)
