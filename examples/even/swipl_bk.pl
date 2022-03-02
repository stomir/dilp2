zero(0, 0).
suk(A, B) :- (integer(A); integer(B)), !, succ(A, B).
suk(A, B) :- when((nonvar(A); nonvar(B)), succ(A, B)).
target(_, _) :- false.