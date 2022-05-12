% To run this code
% clingo aux.lp -c members=12 -c generations=6
% members -- number of people in the family
% generations -- should be less than members, the youngest generation

%The people
people(1..members).

% a person can be male or female
{male(X);female(X)}=1 :- people(X).

%The Edges
{edge(X,Y): people(X),people(Y)}.

%one direction allowed and no self edges
:- edge(X,Y), edge(Y,X).

%The number of incoming edges
inDeg(X,M) :- people(X) , #count{Y: edge(Y,X)} = M.

%The number of outgoing edges
outDeg(X,M):- people(X) , #count{Y: edge(X,Y)} = M.

%only members of the opposite sex can have children together
compatable(X,Y):- female(X),male(Y).
compatable(X,Y):- female(Y),male(X).

%The oldest generation has indegree zeros
gen(X,0):-  inDeg(X,0).

%X is in generation N if one of its parents is in
% genetional N-1
gen(X,N):-  people(X),inDeg(X,2),
            edge(Z,X),gen(Z,M1),
            edge(Y,X),gen(Y,M2),
            Y!=Z,M1>=M2, M1=N-1,
            N<=generations, Z<X,Y<X.

% path between two nodes
path(Y,Y):- people(Y).
path(X,Y):- edge(X,Z), path(Z,Y).

%roughly an equal number of males and females
:-  #count{X: male(X)}=M,#count{X: female(X)}=N, T= |M-N| , T>1.

%onlz for model limitations
:- gen(X,M), female(X), gen(Y,M), male(Y), Y>X.

% the parents of a child must be of a different sex
:- edge(X,Z), edge(Y,Z), X!= Y, not compatable(X,Y).

% How many generations do we want?
:-  #count{X: gen(X,generations)} = 0 .

% Every person is in a unique generation
:-  gen(X,M),gen(X,N), N!=M.

% Every person is in a generation
:- people(X), not gen(X,_).

% Members of the same generation cannot have an edge between them
:-  gen(X,M), gen(Y,M), edge(X,Y).

%Addressing the ethical issues of our study
:- edge(X,Y), edge(X,Z), path(Y,Z), X!=Y, Y!=Z, X!=Z.

%The youngest generation does not have descendants
:- gen(Y,generations), outDeg(Y,M), M>0.

%Every member of the older generations has descendants
:-  gen(Y,W),W<generations, outDeg(Y,M), M=0.

%The oldest generation does not have ancestors
:-  gen(Y,0), outDeg(Y,M), M=0.

%Every person is a descendant of themselves
descendants(X,X,0):- people(X).

%If X is a M-descendant of Y and Y has an edge with Z
% then X is an (M-1)-descendant of Z
descendants(X,Y,M):- M<=generations,edge(Y,Z),W=M-1, descendants(X,Z,W).

%If X is an (M,0)-ancestor of Y then it is an  M-descendant of Y
ancestor(X,Y,M,0):- M<=generations, descendants(X,Y,M).

%If X is a (M,N)-ancestor of Y and Z has an edge with Y
% then X is an (M,N-1)-ancestor of Z
ancestor(X,Y,M,N):- N<=generations,M<=generations, edge(Z,Y),W=N-1, ancestor(X,Z,M,W).

% X is a mother of Y
mother(X,Y):- ancestor(X,Y,0,1), female(X).
% X is a father of Y
father(X,Y):- ancestor(X,Y,0,1), male(X).

% X is a parent of Y
parent(X,Y):- ancestor(X,Y,0,1).

% X is a daughter of Y
daughter(X,Y):- parent(Y,X), female(X).
% X is a son of Y
son(X,Y):- parent(Y,X), male(X).

% X is a child of Y
child(X,Y):- ancestor(Y,X,0,1).
