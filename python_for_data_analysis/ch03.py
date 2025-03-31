# demo of function return multi values
from loguru import logger
import itertools


def demo():
    return 1, 2, 3


a, b, c = demo()

logger.info(f"a={a}, b={b}, c={c}")


# generator example
def gen_demo():
    yield 1
    yield 2
    yield 3


# no code is executed at this point
gen = gen_demo()

for x in gen:
    logger.info(f"x={x}")

# list comp vs generator

list_comp = [x for x in range(10)]
logger.info(f"list_comp={list_comp}")

list_gen = (x for x in range(10))
logger.info(f"list_gen={list_gen}")

# itertools example


def first_letter(name):
    return name[0]


names = ["bob", "joe", "sally", "adam", "jane", "mike", "bill"]

grouped_names = itertools.groupby(names, first_letter)
logger.info(f"grouped_names={grouped_names}")

for letter, names in grouped_names:
    logger.info(f"letter={letter}, names={list(names)}")
