from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from lists([4,3,2,1], empty=False)
        yield 0

    def world2(x) -> Iterable:
        yield from lists([2,3,2,4], empty=False)
        yield 0

class Validate:
    def world1(x) -> Iterable:
        yield from lists([1,2,3,4,5,6,7], empty=False)
        yield 0

class BK:
    def head(a,b):
        return a[0] == b

    def tail(a,b):
        return a[1:] == b or len(a) == 1 and b == 0

class Targets:
    def member(a,b):
        return a in b