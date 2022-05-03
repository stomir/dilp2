from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(16)

class Validate:
    def world1(x) -> Iterable:
        yield from range(31)

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

    def plus2(a,b):
        return a+2 == b

    def plus3(a,b):
        return a+3 == b

class Targets:
    def buzz(a,b):
        if a != b: raise Skip
        return a % 5 == 0