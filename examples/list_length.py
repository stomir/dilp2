from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from lists([3,2,1], empty=False)
        yield 0

    def world2(x) -> Iterable:
        yield from lists([1,2,3], empty=False)
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

    def zero(a,b):
        return a == b and a == 0

    def succ(a,b):
        return a+1 == b

class Targets:
    def length(a,b):
        if (type(a) != tuple and a != 0) or type(b) != int:
            raise Skip
        return (a == 0 and b == 0) or len(a) == b