class Skip(Exception):
    pass

def lists(l):
    for i in range(len(l)+1):
        yield tuple(l[i:])
    yield from l