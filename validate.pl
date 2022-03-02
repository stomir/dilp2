main :- 
    current_prolog_flag(argv, [Directory]),
    format('examples directory ~s\n', [Directory]),
    string_concat(Directory, 'swipl_bk.pl', PathBK),
    string_concat(Directory, 'swipl_val.pl', PathVal),
    Files = [PathBK, PathVal],
    load_files(Files),
    load_input,
    format('loaded files ~s ~s\n', Files),
    format('result: '),
    (
        call_with_depth_limit(validate, 10000, _),
        format('OK\n'),
        halt(0)
     ; 
        format('FAIL\n'),
        halt(1)
    ).

load_input :-
    read(Clause),
    ((Clause=end_of_file, !) ; 
    (assertz(Clause), format('asserted ~w\n', [Clause]), load_input)).


:- set_prolog_flag(verbose, silent).
:- initialization main.