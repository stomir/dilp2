%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp spouse '{"generations":3,"members":7}' spouse '["wife","husband"]' --models 100

% X is the husband of Y (strict)
husband(X,Y):- parent(X,Z),parent(Y,Z),male(X),female(Y).

% X is the wife of Y (strict)
wife(X,Y):- husband(Y,X). 

% X is the spouse of Y 
spouse(X,Y):- husband(X,Y).
spouse(X,Y):- wife(X,Y).

antispouse(X,Y):- not spouse(X,Y), people(X),people(Y).

#show wife/2.
#show husband/2.
#show spouse/2.
#show antispouse/2.
