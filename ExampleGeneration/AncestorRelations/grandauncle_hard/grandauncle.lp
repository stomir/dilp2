%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp grandauncle '{"generations":4,"members":13}' grandauncle '["brother","sister","mother","father"]' --models 100 --postfix hard

% X is the husband of Y (strict)
husband(X,Y):- parent(X,Z),parent(Y,Z),male(X),female(Y).

% X is the wife of Y (strict)
wife(X,Y):- husband(Y,X). 

% X is a sister of Y
sister(X,Y):- parent(Z,X),parent(Z,Y),female(X),X!=Y. 

% X is a brother of Y
brother(X,Y):- parent(Z,X),parent(Z,Y),male(X),X!=Y. 

:- brother(X,Y), parent(X,Z), parent(Y,Z). 
:- sister(X,Y), parent(X,Z), parent(Y,Z). 

% X is a sibling of Y
sibling(X,Y):- sister(X,Y).
sibling(X,Y):- brother(X,Y).

%causes problems without
:- sibling(X,Y), parent(X,Z), parent(Y,Z). 

% X is the spouse of Y 
spouse(X,Y):- husband(X,Y).
spouse(X,Y):- wife(X,Y).

grandparent(X,Y):- parent(X,Z), parent(Z,Y).

grandauncle(X,Y):- grandparent(Z,Y), sibling(X,Z).
antigrandauncle(X,Y):- not grandauncle(X,Y), people(X),people(Y).

#show brother/2.
#show sister/2.
#show mother/2.
#show father/2.
#show grandauncle/2.
#show antigrandauncle/2.
