import fire #type: ignore
import pyparsing as pp #type: ignore
import sys
from typing import *

number = pp.Word(pp.nums + '.')
variable = pp.Word(pp.alphas + pp.nums + '_')
relationship = variable.setResultsName('relationship', listAllMatches=True)
# an argument to a relationship can be either a number or a variable
argument = pp.Word(pp.alphas + pp.nums + '.' + '_')

# arguments are a delimited list of 'argument' surrounded by parenthesis
arguments = (pp.Suppress('(') + pp.delimitedList(argument) +
             pp.Suppress(')')).setResultsName('arguments', listAllMatches=True)

# a fact is composed of a relationship and it's arguments
# (I'm aware it's actually more complicated than this
# it's just a simplifying assumption)
fact = (relationship + arguments).setResultsName('facts', listAllMatches=True)

# a sentence is a fact plus a period
sentence = fact + ':-' + pp.delimitedList(fact) + pp.Optional('.')

def add_edge(edges : Dict[str, Set[str]], a : str, b : str):
    if a not in edges:
        edges[a] = set()
    edges[a].add(b)

def reachable(edges : Dict[str, Set[str]], a : Set[str]) -> Set[str]:
    ret : Set[str] = a
    queue : List[str] = list(a)
    while len(queue) != 0:
        v = queue.pop()
        if v in edges:
            for v2 in edges[v]:
                if v2 not in ret:
                    queue.append(v2)
                    ret.add(v2)
    return ret

def main(show_all : bool = False, target : Set[str] = set(), show_cut_code : bool = False):
    edges : Dict[str, Set[str]] = {}
    code : Dict[str, List[str]] = {}
    target = set(target)
    for line in sys.stdin.readlines():
        parsed = sentence.parseString(line)
        add_edge(edges, parsed[0], parsed[4])
        add_edge(edges, parsed[0], parsed[7])
        print(f"{parsed[0]=} {parsed[4]=} {parsed[7]=}")
        if len(target) == 0:
            target = {parsed[0]}
        if parsed[0] not in code:
            code[parsed[0]] = []
        code[parsed[0]].append(line)

    reached = reachable(edges, target)
    print(f"{target=} {edges=} {reached=}")
    print ('digraph G {')
    for v1, v2s in edges.items():
        if not show_all and v1 not in reached:
            continue
        for v2 in v2s:
            print(f"{v1} -> {v2}")
    print("}")

    if show_cut_code:
        for pred, lines in code.items():
            if pred not in reached:
                continue
            for line in lines:
                print(line)

if __name__ == "__main__":
    fire.Fire(main)
