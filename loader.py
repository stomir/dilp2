import pyparsing as pp #type: ignore
from typing import *
import os.path
import logging

# relationship will refer to 'track' in all of your examples
relationship = pp.Word(pp.alphas).setResultsName('relationship', listAllMatches=True)

number = pp.Word(pp.nums + '.')
variable = pp.Word(pp.alphas)
# an argument to a relationship can be either a number or a variable
argument = number | variable

# arguments are a delimited list of 'argument' surrounded by parenthesis
arguments = (pp.Suppress('(') + pp.delimitedList(argument) +
             pp.Suppress(')')).setResultsName('arguments', listAllMatches=True)

# a fact is composed of a relationship and it's arguments
# (I'm aware it's actually more complicated than this
# it's just a simplifying assumption)
fact = (relationship + arguments).setResultsName('facts', listAllMatches=True)

# a sentence is a fact plus a period
sentence = fact + pp.Suppress('.')

# self explanatory
prolog_sentences = pp.OneOrMore(sentence)

class Problem(NamedTuple):
    predicates : Dict[str, int]
    bk : Set[int]
    main : int
    invented : Set[int]

class World(NamedTuple):
    atoms : Dict[str, int]
    facts : Sequence[Tuple[int,int,int]]
    positive : Sequence[Tuple[int,int,int]]
    negative : Sequence[Tuple[int,int,int]]

A = TypeVar('A')
B = TypeVar('B')
def rev_dict(d : Dict[A, B]) -> Dict[B, A]:
    return dict((v, k) for k, v in d.items())

def load_facts(filename : str) -> Iterable[Tuple[str,str,str]]:
    with open(filename) as f:
        data = prolog_sentences.parseString(f.read().replace('\n', ' '))
    for idx in range(len(data['facts'])):
        fact = data['facts'][idx]
        predicate : str = data['relationship'][idx]
        args : Sequence[str] = data['arguments'][idx]
        if len(args) != 2:
            raise ValueError(f'predicate arity other than 2, {fact=} {predicate=} {args=}')
        logging.debug(f'{(predicate, args[0], args[1])=}')
        yield (predicate, args[0], args[1])

def indexify(data : Iterable[Tuple[str,str,str]], preds : Dict[str, int], atoms : Dict[str, int]) -> Iterable[Tuple[int,int,int]]:
    return ((preds[head], atoms[arg1], atoms[arg2]) for head, arg1, arg2 in data)

def load_problem(dir : str, invented_count : int) -> Problem:
    facts = list(load_facts(os.path.join(dir, 'facts.dilp')))
    examples = list(load_facts(os.path.join(dir, 'positive.dilp'))) \
                + list(load_facts(os.path.join(dir, 'negative.dilp')))

    bk = set(f[0] for f in facts)
    targets = set(f[0] for f in examples)
    target = next(iter(targets))
    if len(targets) != 1:
        raise ValueError('number of target predicates other than 1')

    all_preds : Dict[str, int] = {target : 0}
    all_preds.update(zip(bk, range(1,1+len(bk))))

    invented = list(range(1+len(bk), 1+len(bk)+invented_count))
    all_preds.update((f'inv_{i}', n) for i, n in enumerate(invented))

    logging.info(f'loaded problem from {dir}')
    return Problem(
        predicates = all_preds,
        bk = set(all_preds[p] for p in bk),
        main = all_preds[target],
        invented = set(invented)
    )
    
def load_world(dir : str, problem : Problem) -> World:
    facts = list(load_facts(os.path.join(dir, 'facts.dilp')))
    atoms_set : Set[str] = set.union(*(set(f[1:]) for f in facts))
    logging.debug(f'{dir=} {atoms_set=} {facts=}')
    atoms = dict(zip(atoms_set, range(len(atoms_set))))
    positive = list(indexify(load_facts(os.path.join(dir, 'positive.dilp')), problem.predicates, atoms))
    negative = list(indexify(load_facts(os.path.join(dir, 'negative.dilp')), problem.predicates, atoms))
    assert all(head == problem.main for head, _, _ in positive)
    assert all(head == problem.main for head, _, _ in negative)

    logging.info(f'loaded world from {dir}')
    return World(
        atoms = atoms,
        facts = list(indexify(facts, problem.predicates, atoms)),
        positive = positive,
        negative = negative,
    )
