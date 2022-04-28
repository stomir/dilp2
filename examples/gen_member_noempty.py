from genutils import *
from typing import *

def wurl(n : int) -> Iterable:
    yield from lists(''.join(chr(ord('a') + i) for i in range(n)))

class Train:
    def world1(x) -> Iterable:
        yield from wurl(5)

class Validate:
    def world1(x) -> Iterable:
        yield from wurl(10)

class BK:
#s    def empty(a,b):
#        return a == b and len(a) == 0

    def head(a,b):
        return type(a) is tuple and a[0] == b

    def tail(a,b):
        return a[1:] == b and a != ()

class Targets:
    def member(a,b):
        if type(b) is not tuple or type(a) is not str:
            raise Skip
        return a in b
