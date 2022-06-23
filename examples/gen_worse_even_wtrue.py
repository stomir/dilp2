from genutils import *
from typing import *

class Train:
    def world1(x) -> Iterable:
        yield from range(10)
        yield 'a'

class Validate:
    def world1(x) -> Iterable:
        yield from range(20)
        yield 'a'

class BK:
    def zero(a,b):
        return a == 0 and b == 0

    def succ(a,b):
        return a+1 == b

    def true(a,b):
        return True

    def false(a,b):
        #return a == '_' and b == '_'
        return a == 'a' and b == 'a'

    
    # def false2(a,b):
    #     #return a == '_' and b == '_'
    #     return a == 'a' and b == 'a'

    
    # def false3(a,b):
    #     #return a == '_' and b == '_'
    #     return a == 'a' and b == 'a'

class Targets:
    def even(a,b):
        if a != b: raise Skip
        return a % 2 == 0