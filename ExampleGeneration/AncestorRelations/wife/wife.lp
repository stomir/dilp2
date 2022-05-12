%Aux.lp: mother,father,parent,daughter,son,child

% python genTask.py aux.lp wife '{"generations":3,"members":7}' wife '["mother","father"]' --models 100


% X is the wife of Y (strict)
wife(X,Y):- mother(X,Z),father(Y,Z). 


antiwife(X,Y):- not wife(X,Y), people(X),people(Y).

#show mother/2.
#show father/2.
#show wife/2.
#show antiwife/2.
