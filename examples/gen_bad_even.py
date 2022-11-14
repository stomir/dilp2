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

    def true1(a,b):
        return True

    def true2(a,b):
        return True

class Targets:
    def even(a,b):
        if a != b: raise Skip
        return a % 2 == 0