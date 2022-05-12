%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp auncle '{"generations":3,"members":9}' auncle '["sister","brother","parent"]' --models 100 --postfix med

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

auncle(X,Y):- parent(Z,Y), sibling(X,Z).
antiauncle(X,Y):- not auncle(X,Y), people(X),people(Y).

#show sister/2.
#show brother/2.
#show parent/2.
#show auncle/2.
#show antiauncle/2.
