% python genTask.py aux.lp sister '{"generations":3,"members":7}' sister '["mother","father","daughter","son"]' --models 100 --postfix easy

% X is a Sister of Y
sister(X,Y):- parent(Z,X),parent(Z,Y),daughter(X,Z).

antisister(X,Y):- not sister(X,Y), people(X),people(Y).

#show mother/2.
#show father/2.
#show daughter/2.
#show sister/2.
#show antisister/2.
