from genutils import *
from typing import *

class Wurl(NamedTuple):
    colors : Dict[str, str]
    edges : Dict[str, Set[str]]

red = 'red'
green = 'green'
blue = 'blue'
colors = {red, green, blue}

class Train:
    def world1(x) -> Iterable:
        x.data = Wurl(
            colors = {},
            edges = {'a': {'b'}, 'b':{'c','d'}, 'c':{'a'}, 'd':{'e','f'}, 'e':{'f'}, 'f':{'e'}},
        )
        yield from x.data.edges.keys()
    
    def world2(x) -> Iterable:
        x.data = Wurl(
            colors = {},
            edges = {'a': {'b'}, 'b': set(), 'c':{'d'}, 'd':{'e'},'e':{'c'}}
        )
        yield from x.data.edges.keys()

class Validate:
    
    def world1(x) -> Iterable:
        x.data = Wurl(
            colors = {},
            edges = {'a': {'b'}, 'b': {'a'}, 'c':set(), 'd':{'c'}}
        )
        yield from x.data.edges.keys()

class BK:
    def edge(a,b):
        return b in data.edges[a]

def reachable(data : Dict[str, Set[str]], a : str) -> Set[str]:
    ret = set()
    queue = list(data[a])
    while len(queue) > 0:
        v = queue.pop()
        ret.add(v)
        for e in data[v]:
            if e in ret:
                continue
            queue.append(e)
    return ret

class Targets:
    def cyclic(a,b):
        if a != b:
            raise Skip
        return a in reachable(data.edges, a)
        #return len(set(BK.color(e, red) for e in data.edges[a])) > 1 and BK.color(a, green) #any(BK.color(e, green) for e in data.edges[a])