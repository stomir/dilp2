%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp niece '{"generations":3,"members":9}' niece '["brother","sister","daughter","mother","father"]' --models 100 --postfix hard


% X is the wife of Y (strict)
% X is a Sister of Y
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

niece(X,Y):- daughter(X,Z), sibling(Z,W), parent(W,Y).

antiniece(X,Y):- not niece(X,Y), people(X),people(Y).

#show niece/2.
#show brother/2.
#show sister/2.
#show daughter/2.
#show parent/2.
#show mother/2.
#show father/2.
#show antiniece/2.
