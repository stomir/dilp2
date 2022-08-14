from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(5)

class Validate:
    def world1(x) -> Iterable:
        yield from range(10)

class BK:
    # def zero(a,b):
    #     return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

class Targets:
    def gt(a,b):
        return a > b