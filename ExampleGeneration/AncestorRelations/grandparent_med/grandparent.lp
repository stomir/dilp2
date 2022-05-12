%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp grandparent '{"generations":3,"members":9}' grandparent '["mother","father"]' --models 100 --postfix med

grandparent(X,Y):- parent(X,Z), parent(Z,Y).
antigrandparent(X,Y):- not grandparent(X,Y), people(X),people(Y).

#show mother/2.
#show father/2.
#show grandparent/2.
#show antigrandparent/2.
