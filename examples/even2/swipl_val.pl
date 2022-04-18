zero(0, 0).
suk(A, B) :- (integer(A); integer(B)), !, succ(A, B).
suk(A, B) :- when((nonvar(A); nonvar(B)), succ(A, B)).

target_predicate(target).
validate :-
    target(0, 0),
    target(12, 12),
    target(14, 14),
    target(16, 16),
    target(18, 18),
    target(20, 20),
    target(32, 32),
    \+target(1, 1),
    \+target(31, 31),
    \+target(11, 11).