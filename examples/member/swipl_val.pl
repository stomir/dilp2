head([H|_], H).
tail([_|T], T).
empty([], []).

target_predicate(member).
validate :-
    member([a,b], a),
    member([a,b,c,d,e,f,g,h,i,j], i),
    member([a], a),
    \+member([], _),
    \+mem([a,b,c], d).