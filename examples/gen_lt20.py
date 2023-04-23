from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(20)

class Validate:
    def world1(x) -> Iterable:
        yield from range(30)

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

class Targets:
    def lt(a,b):
        return a < b