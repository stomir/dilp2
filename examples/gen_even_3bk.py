from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(10)

class Validate:
    def world1(x) -> Iterable:
        yield from range(20)

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

    def zero2(a,b):
        return a == 0 and b == 0

    def succ2(a,b):
        return a+1 == b

    def zero3(a,b):
        return a == 0 and b == 0

    def succ3(a,b):
        return a+1 == b

class Targets:
    def even(a,b):
        if a != b: raise Skip
        return a % 2 == 0