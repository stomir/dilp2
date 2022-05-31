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
            edges = {'a': {'b'}, 'b':{'c','a'}, 'c':{'d'}, 'd':set()},
        )
        yield from x.data.edges.keys()

    def world2(x) -> Iterable:
        x.data = Wurl(
            colors = {},
            edges = {'a': {'b'}, 'b':{'c','a'}, 'c':{'d'}, 'd':{'e'}, 'e':{'e'},'f':{'g'},'g':{'h'},'h':set()},
        )
        yield from x.data.edges.keys()

class Validate:
    def world1(x) -> Iterable:
        x.data = Wurl(
            colors = {},
            edges = {'a': {'b'}, 'b': {'c','d'}, 'c':{'a','b', 'e'}, 'd': {'e','b','h','f'}, 'e':set(), 'f':{'a','h','c'}, 'g': {'g'}, 'h':{'b','d','g'}, 'i':{'h','b'}}
        )
        yield from 'abcdefghi'


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
    def conn(a,b):
        return b in reachable(data.edges, a)
        #return len(set(BK.color(e, red) for e in data.edges[a])) > 1 and BK.color(a, green) #any(BK.color(e, green) for e in data.edges[a])