import fire #type: ignore
from typing import *
import inspect
import os
import logging
from genutils import Skip

def repr(atom):
    if type(atom) is tuple:
        return 'list_' + ''.join(str(a) for a in atom)
    return str(atom)

def call(function : Callable, a1, a2) -> Optional[bool]:
    try:
        if function(a1, a2):
            return True
    except Skip:
        return None
    except TypeError:
        pass
    except IndexError:
        pass
    return False

def facts(function : Callable, name : str, atoms : set) -> Iterable[str]:
    for a1 in atoms:
        for a2 in atoms:
            if call(function, a1, a2):
                yield (f"{name}({repr(a1)}, {repr(a2)})")

def preds(obj, atoms : set) -> Iterable[str]:
    for name, function in inspect.getmembers(obj, inspect.isfunction):
        yield from facts(function, name, atoms)

def gen_world(atoms : set, outdir : str, module):
    os.mkdir(outdir)
    with open(os.path.join(outdir, 'facts.dilp'), 'w') as facts_file:
        for name, func in inspect.getmembers(module.BK, inspect.isfunction):
            for a1 in atoms:
                for a2 in atoms:
                    if call(func, a1, a2):
                        logging.info(f"fact: {name}({repr(a1)}, {repr(a2)})")
                        facts_file.write(f"{name}({repr(a1)},{repr(a2)})\n")

    with open(os.path.join(outdir, 'positive.dilp'), 'w') as true_file:
        with open(os.path.join(outdir, 'negative.dilp'), 'w') as false_file:
            for name, func in inspect.getmembers(module.Targets, inspect.isfunction):
                for a1 in atoms:
                    for a2 in atoms:
                        result = call(func, a1, a2)
                        if result is None:
                            continue
                        file = true_file if result else false_file
                        file.write(f"{name}({repr(a1)},{repr(a2)})\n")
                        logging.info(f"training {name}({repr(a1)},{repr(a2)}) {result=}")

def main(name : str, outdir : Optional[str] = None, info : bool = True):
    if info:
        logging.getLogger().setLevel(logging.INFO)

    if name[-3:] == '.py':
        name = name[:-3]
    module = __import__(name)
    if outdir is None:
        outdir = name

    os.mkdir(outdir)

    for name, func in inspect.getmembers(module.Train, inspect.isfunction):
        atoms = set(func(None))
        dirname = os.path.join(outdir, 'train_'+name)
        logging.info(f"=== training world {name} {dirname=} {atoms=}")
        gen_world(atoms, dirname, module)

    for name, func in inspect.getmembers(module.Validate, inspect.isfunction):
        atoms = set(func(None))
        dirname = os.path.join(outdir, 'val_'+name)
        logging.info(f"=== validation world {name} {dirname=} {atoms=}")
        gen_world(atoms, dirname, module)



if __name__ == "__main__":
    fire.Fire(main)