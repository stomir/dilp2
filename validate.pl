main :- 
    current_prolog_flag(argv, [Directory|ArgVs]),
    format('examples directory ~s\n', [Directory]),
    string_concat(Directory, 'swipl_val.pl', PathVal),
    Files = [PathVal],
    load_files(Files),
    target_predicate(Name),
    redefine_system_predicate(Name/2),
    NamedTerm =.. [Name,_,_],
    assertz(':-'(NamedTerm, false)),
    load_input,
    format('loaded files ~w\n', [Files]),
    (
        ArgVs=[NS|_],
        atom_number(NS, N)
        ;
        N = 1000
    ),
    format('steps: ~d files: ~w\n', [N,Files]),
    format('result: '),
    (
        call_with_inference_limit(validate, N, _),
        format('OK\n'),
        halt(0)
     ; 
        format('FAIL\n'),
        halt(1)
    ).

load_input :-
    read(Clause),
    (Clause=end_of_file, ! ; 
    assertz(Clause), load_input).


:- set_prolog_flag(verbose, silent).
:- initialization main.