from genutils import *
from typing import *

class Wurl(NamedTuple):
    mom : Dict[str, str]
    dad : Dict[str, str]

    def names(self) -> Set[str]:
        s = set()
        s |= set(self.mom.keys())
        s |= set(self.mom.values())
        s |= set(self.dad.keys())
        s |= set(self.dad.values())
        return s

class Train:
    def world1(x) -> Iterable:
        x.data = Wurl(
            mom = {'a' : 'i', 'g' : 'c', 'f' : 'c', 'h' : 'f'},
            dad = {'b' : 'a', 'c' : 'a', 'd' : 'b', 'e' : 'b'}
        )
        return x.data.names()

class Validate:
    
    def world1(x) -> Iterable:
        x.data = Wurl(
            mom = {'a' : 'b', 'b' : 'c', 'c' : 'd', 'e' : 'f'},
            dad = {'a' : 'e', 'e' : 'g', 'b' : 'h', 'h' : 'j'}
        )
        return x.data.names()
class BK:
    def mom(a,b):
        return data.mom[b] == a

    def dad(a,b):
        return data.dad[b] == a

def parents(a):
    ret = set()
    if a in data.mom:
        yield data.mom[a]
    if a in data.dad:
        yield data.dad[a]

def grandparents(a):
    for p in parents(a):
        yield from parents(p)

class Targets:
    def grandparent(a,b):
        return a in set(grandparents(b))