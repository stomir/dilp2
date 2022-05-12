%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp grandparent '{"generations":3,"members":9}' grandparent '["parent"]' --models 100 --postfix easy



grandparent(X,Y):- parent(X,Z), parent(Z,Y).
antigrandparent(X,Y):- not grandparent(X,Y), people(X),people(Y).

#show parent/2.
#show grandparent/2.
#show antigrandparent/2.
