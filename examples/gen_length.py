from genutils import *

class Train:
    def world1(x):
        yield from range(7)
        yield from lists('abcdefg')

class Validate:
    def world1(x):
        yield from range(10)
        yield from lists('abcdefghij')

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