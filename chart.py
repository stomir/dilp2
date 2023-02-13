import fire #type: ignore
import natsort
from typing import *
import os
import numpy
import matplotlib.pyplot #type: ignore

def get_n(s : str) -> int:
    return int(s.split(' ')[-1].split('/')[0])

def which(line : str, things : Iterable[str]) -> Optional[str]:
    for thing in sorted(things, key=len, reverse=True):
        if thing in line:
            return thing
    return None

def count(lines : Iterable[str], things : Iterable[str]) -> Dict[str, int]:
    ret = {thing : 0 for thing in things}
    for line in lines:
        w = which(line, things)
        if w is not None:
            ret[w] += 1
    return ret


def main(*files, idx_key : Optional[int] = None, output : str = "chart.png") -> None:

    if idx_key is None:
        a = natsort.natsort_key(files[0])
        b = natsort.natsort_key(files[1])
        idx_key = next(i for i, (aa, bb) in enumerate(zip(a,b)) if aa != bb)

    outcomes = ["OK", "FUZZY", "OVERFIT", "FUZZY OVERFIT", "FAIL"]
    colors = ["green", "yellow", "orange", "purple", "red"]

    x : Dict[str, List[int]] = {o : [] for o in outcomes}
    labels : List[object] = []

    for key, file in sorted(((natsort.natsort_key(f), f) for f in files)):
        lines = open(os.path.join(file, 'report'), 'r').readlines()
        for o, n in count(lines, outcomes).items():
            x[o].append(n)
        labels.append(key[idx_key])

    print(f"{labels=}\n{x=}")
    matplotlib.pyplot.stackplot(labels, [x[o] for o in outcomes], colors=colors, alpha=0.5, labels=outcomes)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig(output)





if __name__ == "__main__":
    fire.Fire(main)