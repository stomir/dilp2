member(A, B) :- head(A, B).
member(A, B) :- tail(A, C), member(C, B).