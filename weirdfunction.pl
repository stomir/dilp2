
zero(0,X) :- X is 0.
zero(X,0) :- 0 is 0.
target(A,B,N) :- N>0,print(t1(A,B,N)),succ(A,C),succ(C,B).
target(A,B,N) :- N>0,M is N-1,print(t2(A,B,N)), inv_0(C,B,M),target(C,A,M).
inv_0(A,B,N) :- N>0,print(i1(A,B,N)),zero(A,_),zero(B,A).
inv_0(A,B,N) :- N>0,M is N-1,print(i2(A,B,N)),target(C,A,M),inv_0(C,B,M).



