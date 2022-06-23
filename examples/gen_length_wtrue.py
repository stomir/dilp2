from genutils import *
from typing import *

def wurl(n : int) -> Iterable:
    yield from range(n+1)
    

class Train:
    def world1(x) -> Iterable:
        yield from range(4)
        yield from lists('aaa')

class Validate:
    def world1(x) -> Iterable:
        yield from range(7)
        yield from lists('abcaca')

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

    def head(a,b):
        return type(a) is tuple and a[0] == b

    def true(a,b):
        return True

    def tail(a,b):
        if b == 0:
            b = ()
        elif b == ():
            b = None
        return a[1:] == b and a != ()

class Targets:
    def length(a,b):
        if a == 0 and b == 0:
            return True
        if type(a) is not tuple or type(b) is not int or a == ():
            raise Skip
        return len(a) == b