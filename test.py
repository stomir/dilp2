def test(a : int) -> int:
    for i in range(10):
        a += a
    return a