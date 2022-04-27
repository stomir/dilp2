from genutils import *
from typing import *

def wurl(n : int) -> Iterable:
    yield from range(n+1)
    yield from lists(''.join(chr(ord('a') + i) for i in range(n)))

class Train:
    def world1(x) -> Iterable:
        yield from wurl(5)

class Validate:
    def world1(x) -> Iterable:
        yield from wurl(10)

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

    def head(a,b):
        return type(a) is tuple and a[0] == b

    def tail(a,b):
        return a[1:] == b and a != ()

class Targets:
    def length(a,b):
        if type(a) is not tuple or type(b) is not int:
            raise Skip
        return len(a) == b