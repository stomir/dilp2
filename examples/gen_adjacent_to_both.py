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
            colors = {'a': red, 'b': green,'c': green,'d': red, 'e':red},
            edges = {'a': {'b', 'c'}, 'b': {'c','d'}, 'c':{'d','e'}, 'd': {'e','a'}, 'e':{'a', 'b'}}
        )
        yield from 'abcde'
        yield red
        yield green

    def world2(x) -> Iterable:
        x.data = Wurl(
            colors = {'a': red, 'b': green,'c': green,'d': green, 'e':green},
            edges = {'a': {'b', 'c','d'}, 'b': {'c','d','e'}, 'c':{'d','e','a'}, 'd': {'e','a','b'}, 'e':{'a', 'b','c'}}
        )
        yield from 'abcde'
        yield red
        yield green

    def world3(x) -> Iterable:
        x.data = Wurl(
            colors = {'a': red, 'b': red,'c': green,'d': green, 'e':green},
            edges = {'a': {'b', 'c'}, 'b': {'c','d'}, 'c':{'d','e'}, 'd': {'e','a'}, 'e':{'a', 'b'}}
        )
        yield from 'abcde'
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

    def neq_color(a,b):
        return a in colors and b in colors and a != b

    def edge(a,b):
        return b in data.edges[a]

class Targets:
    def adj_to_both(a,b):
        if a != b or b in colors:
            raise Skip
        return any(data.colors[e] == data.colors[a] for e in data.edges[a]) and any(data.colors[e] != data.colors[a] for e in data.edges[a])
        #return len(set(BK.color(e, red) for e in data.edges[a])) > 1 and BK.color(a, green) #any(BK.color(e, green) for e in data.edges[a])