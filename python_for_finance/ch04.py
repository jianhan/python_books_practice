import numpy as np
from loguru import logger

# initialize nd array first then populate

c = np.zeros((3, 2), dtype=int, order="C")

logger.info(f"zeros values in c is {c}")

c = np.ones((2, 3, 4), dtype=int)
logger.info(f"ones values in c is {c}")

c = np.ones_like(c, dtype="f16")
logger.info(f"ones like values in c is {c}")

size = c.size
logger.info(f"size of c is {size}")

itemsize = c.itemsize
logger.info(f"itemsize of c is {itemsize}")

ndim = c.ndim
logger.info(f"ndim of c is {ndim}")

shape = c.shape
logger.info(f"shape of c is {shape}")

dtype = c.dtype
logger.info(f"dtype of c is {dtype}")

g = np.arange(10)
logger.info(f"arange values in g is {g}")
logger.info(f"shape of g is {g.shape}")

reshaped_g = g.reshape((2, 5))
logger.info(f"reshape values in reshaped_g is {reshaped_g}")
transposed_g = reshaped_g.transpose()
logger.info(f"transpose values in transposed_g is {transposed_g}")

# resize examples

g = np.arange(15)
resized_g = np.resize(g, (3, 5))
logger.info(f"resize values in g is {reshaped_g}")

h = np.arange(15)
greater_than_10 = h > 10
logger.info(f"greater_than_10 values in h is {greater_than_10}")

# use as data selection
greater_than_5 = h[h > 5]
logger.info(f"greater_than_5 values in h is {greater_than_5}")

r = np.arange(12).reshape((4, 3))
logger.info(f"r values in h is {r}")
s = np.arange(12).reshape((4, 3)) * 0.5
logger.info(f"s values in h is {s}")

r_plus_s = r + s
logger.info(f"r_plus_s values in h is {r_plus_s}")

s = np.arange(0, 12, 4)
r_plus_s = r + s
logger.info(f"r values in h is {r}")
logger.info(f"s values in s is {s}")
logger.info(f"r_plus_s values in h is {r_plus_s}")

s = np.arange(0, 12, 3)

# r_plus_s = r + s  # throws error
r_plus_s = r.transpose() + s
