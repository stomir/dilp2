%elements of the list
elements(1..ele).

%lists
list(X,Y) :- X=0..nli, W=(ele+1)+X*(li+1), V = (ele+1+li)+X*(li+1), Y=W..V.

%largest list
full(X,Y) :- X=0..nli, Y=(ele+1)+X*(li+1).

%empty list
emp(X,Y):- X=0..nli, Y=(ele+1+li)+X*(li+1).

% A list X can only have Y as a tail1 if X<Y
{tail1(X,Y):list(W,X), X<Y} <= 1 :- list(W,Y).

% Every list with a tail1 must have a head
{head(X,Y): elements(Y)} = 1  :- tail1(X,_).

%elements can only occur in order
{head(X,W): elements(W),head(Y,Z), W<Z} = 1  :- tail1(X,Y), not emp(_,Y).

%elements can only occur in order
{head(X,W): elements(W)} = 1  :- tail1(X,Y), emp(_,Y).

%emp must be in the tail1 of some list
{tail1(Y,X): list(W,Y)}=1:- emp(W,X).

%the full list must have some list as its tail1.
{tail1(X,Y): list(W,Y)}=1:- full(W,X).

:- tail1(X,Y), list(W,X), list(V,Y), W!=V.

% The emp list is always on a path
path(Y,Y,W):- emp(W,Y).

% paths connected by the tail relation
path(X,Y,W):- tail1(X,Z), path(Z,Y,W).

%every sublist must have a path to emp
:- tail1(X,Y), list(V,X),list(V,Y),not path(Y,W,V), emp(V,W).

% Used list must have a front
:- list(0,X),tail1(X,_), not front(X,_).

% Y is the front of X
:- full(1,Y),full(0,X),not front(X,Y).


%Defining the front predicate
front(X,Y):- head(X,_),tail1(X,W),
             emp(0,W), emp(1,Y).
front(X,Y):- list(0,X),list(1,Y),
             head(X,W),head(Y,W),
             tail1(X,V),tail1(Y,R),
             front(V,R).

%joining empty predicates
empty(Y,Y) :- #max{X :emp(_,X)} =Y.

%turns tail1 into tail unless tail1(X,[])
tail(X,Y) :- tail1(X,Y), not emp(_,Y).

%turns tail1 into tail if tail1(X,[])
tail(X,Z) :- tail1(X,Y), emp(_,Y), empty(Z,Z).

%Negative examples

antifront(X,Y):- not front(X,Y), head(X,_), head(Y,_).

antifront(X,Y):- not front(X,Y), empty(X,_), head(Y,_).

antifront(X,Y):- not front(X,Y), head(X,_), empty(Y,_).

#show tail/2.
#show head/2.
#show empty/2.
#show front/2.
#show antifront/2.
