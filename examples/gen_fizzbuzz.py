from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(14)

class Validate:
    def world1(x) -> Iterable:
        yield from range(28)

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

class Targets:
    def fizz(a,b):
        if a != b: raise Skip
        return a % 3 == 0

    def buzz(a,b):
        if a != b: raise Skip
        return a % 3 == 0