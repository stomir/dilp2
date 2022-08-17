import pyparsing as pp #type: ignore
from typing import *
import os.path
import logging
from enum import IntEnum

number = pp.Word(pp.nums + '.')
variable = pp.Word(pp.alphas + pp.nums + '_')
relationship = variable.setResultsName('relationship', listAllMatches=True)
argument = pp.Word(pp.alphas + pp.nums + '.' + '_')
arguments = (pp.Suppress('(') + pp.delimitedList(argument) +
             pp.Suppress(')')).setResultsName('arguments', listAllMatches=True)

fact = (relationship + arguments).setResultsName('facts', listAllMatches=True)

sentence = fact + pp.Optional(pp.Suppress('.'))

class TargetType(IntEnum):
    POSITIVE = 1
    NEGATIVE = 0

class Problem(NamedTuple):
    predicate_number : Dict[str, int]
    predicate_name : Dict[int, str]
    bk : Set[int]
    #main : int
    targets : Set[int]
    invented : Set[int]
    types : Dict[int, List[Optional[str]]]
    all_types : Set[str]

class World(NamedTuple):
    atoms : Dict[str, int]
    dir : str
    train : bool
    facts : Sequence[Tuple[int,int,int]]
    positive : Sequence[Tuple[int,int,int]]
    negative : Sequence[Tuple[int,int,int]]

A = TypeVar('A')
B = TypeVar('B')
def rev_dict(d : Dict[A, B]) -> Dict[B, A]:
    return dict((v, k) for k, v in d.items())

def load_facts(filename : str) -> Iterable[Tuple[str,str,str]]:
    logging.debug(f'loading facts from {filename}')
    lines = (stripped for line in open(filename).readlines() if (stripped := line.strip(' \t\n')) != '')   
    for line in lines:
        logging.debug(f'loading fact {line=}')
        data = sentence.parseString(line)
        predicate : str = data[0]
        args : Sequence[str] = list(data[1:])
        if len(args) != 2:
            raise ValueError(f'predicate arity other than 2, {data=} {predicate=} {args=}')
        logging.debug(f'fact loaded {(predicate, args[0], args[1])}')
        yield (predicate, args[0], args[1])

def indexify(data : Iterable[Tuple[str,str,str]], preds : Dict[str, int], atoms : Dict[str, int]) -> Iterable[Tuple[int,int,int]]:
    return ((preds[head], atoms[arg1], atoms[arg2]) for head, arg1, arg2 in data if arg1 != '_' and arg2 != '_')

def load_problem(dir : str, invented_count : int) -> Problem:
    facts = list(load_facts(os.path.join(dir, 'facts.dilp')))
    examples = list(load_facts(os.path.join(dir, 'positive.dilp'))) \
                + list(load_facts(os.path.join(dir, 'negative.dilp')))

    bk = set(f[0] for f in facts)
    targets = set(f[0] for f in examples)

    all_preds : Dict[str, int] = {}
    inv_names = [f'inv{i}' for i in range(invented_count)]
    all_preds.update(zip(list(sorted(bk)) + list(sorted(targets)) + inv_names, range(len(targets)+len(bk)+invented_count)))

    types : Dict[int, Sequence[str]] = {}
    all_types : Set[str] = set()
    if (os.path.isfile(os.path.join(dir, '..', 'types'))):
        for pred, type1, type2 in load_facts(os.path.join(dir, '..', 'types')):
            types[all_preds[pred]] = [type1, type2, None]
            all_types.add(type1)
            all_types.add(type2)

    logging.info(f'loaded problem from {dir}')
    return Problem(
        predicate_number = all_preds,
        predicate_name = dict((idx, name) for name, idx in all_preds.items()),
        bk = set(all_preds[p] for p in bk),
        targets = set(all_preds[p] for p in targets),
        invented = set(all_preds[p] for p in inv_names),
        types = types,
        all_types = all_types,
    )
    
def load_world(dir : str, problem : Problem, train : bool) -> World:
    logging.debug(f'loading world from {dir}')
    facts = list(load_facts(os.path.join(dir, 'facts.dilp')))
    atoms_set : Set[str] = set.union(*(set(f[1:]) for f in facts)) - {'_'}
    logging.debug(f'{dir=} {atoms_set=} {facts=}')
    atoms = dict(zip(atoms_set, range(len(atoms_set))))
    positive = list(indexify(load_facts(os.path.join(dir, 'positive.dilp')), problem.predicate_number, atoms))
    negative = list(indexify(load_facts(os.path.join(dir, 'negative.dilp')), problem.predicate_number, atoms))
    assert all(head in problem.targets for head, _, _ in positive)
    assert all(head in problem.targets for head, _, _ in negative)


    logging.info(f'loaded world from {dir}')
    return World(
        atoms = atoms,
        facts = list(indexify(facts, problem.predicate_number, atoms)),
        positive = positive,
        negative = negative,
        dir = dir,
        train = train,
    )
