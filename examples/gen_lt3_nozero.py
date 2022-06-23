from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(4)

class Validate:
    def world1(x) -> Iterable:
        yield from range(10)

class BK:
    def succ(a,b):
        return a+1 == b

class Targets:
    def lt(a,b):
        return a < b