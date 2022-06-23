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

class Targets:
    def fizz(a,b):
        if a != b: raise Skip
        return a % 3 == 0
    
    def fizz2(a,b):
        if a != b: raise Skip
        return a % 3 == 0

    def fizz3(a,b):
        if a != b: raise Skip
        return a % 3 == 0