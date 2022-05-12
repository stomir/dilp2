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
            colors = {'a': green, 'b': red,'c': green,'d': green, 'e':red, 'f':red},
            edges = {'a': {'b'}, 'b': {'c','d'}, 'c':{'e','e'}, 'd': set(), 'e':{'f'}, 'f':set()}

        )
        yield from x.data.colors.keys()
        yield red
        yield green

    def world2(x) -> Iterable:
        x.data = Wurl(
            colors = {'a': green, 'b': green,'c': red,'d': green, 'e':green, 'f':red},
            edges = {'a': {'b'}, 'b': {'a'}, 'c':{'b'}, 'd': {'c'}, 'e':{'d'}, 'f':{'c'}}

        )
        yield from x.data.colors.keys()
        yield red
        yield green

class Validate:
    def world1(x) -> Iterable:
        x.data = Wurl(
            colors = {'a': red, 'b': green,'c': red,'d': green, 'e':red, 'f':red, 'g':green, 'h':red},
            edges = {'a': {'b'}, 'b': {'c','d'}, 'c':{'a','b', 'e'}, 'd': {'e','b','h','f'}, 'e':set(), 'f':{'a','h','c'}, 'g': {'g'}, 'h':{'b','d','g'}, 'i':{'h','b'}}
        )
        yield from 'abcdefghi'
        yield red
        yield green


class BK:
    def color(a,b):
        return data.colors[a] == b

    def edge(a,b):
        return b in data.edges[a]

class Targets:
    def wrong(a,b):
        if a != b or b in colors:
            raise Skip
        nodes = data.colors.keys()
        return any(b in data.edges[a] and data.colors[a] == data.colors[b] for b in nodes)
        #return len(set(BK.color(e, red) for e in data.edges[a])) > 1 and BK.color(a, green) #any(BK.color(e, green) for e in data.edges[a])