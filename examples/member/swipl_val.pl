member :- false.
validate :-
    member([a,b], a),
    member([a,b,c,d,e,f,g,h,i,j], i),
    member([a], a),
    \+member([], _),
    \+member([a,b,c], d).