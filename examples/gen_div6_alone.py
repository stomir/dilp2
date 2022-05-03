from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(19)

class Validate:
    def world1(x) -> Iterable:
        yield from range(31)

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

class Targets:
    def div6(a,b):
        if a != b: raise Skip
        return a % 6 == 0