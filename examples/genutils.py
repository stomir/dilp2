class Skip(Exception):
    pass

def lists(l, empty : bool = True):
    for i in range(len(l)+(1 if empty else 0)):
        yield tuple(l[i:])
    yield from l
