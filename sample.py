import fire #type: ignore
import dilp
import torch
from tqdm import tqdm #type: ignore
import pyparsing as pp #type: ignore
import torch
import logging
from core import Term, Atom
from typing import *

# relationship will refer to 'track' in all of your examples
relationship = pp.Word(pp.alphas).setResultsName('relationship', listAllMatches=True)

number = pp.Word(pp.nums + '.')
variable = pp.Word(pp.alphas)
# an argument to a relationship can be either a number or a variable
argument = number | variable

# arguments are a delimited list of 'argument' surrounded by parenthesis
arguments = (pp.Suppress('(') + pp.delimitedList(argument) +
             pp.Suppress(')')).setResultsName('arguments', listAllMatches=True)

# a fact is composed of a relationship and it's arguments
# (I'm aware it's actually more complicated than this
# it's just a simplifying assumption)
fact = (relationship + arguments).setResultsName('facts', listAllMatches=True)

# a sentence is a fact plus a period
sentence = fact + pp.Suppress('.')

# self explanatory
prolog_sentences = pp.OneOrMore(sentence)

def revDict(d):
    return {value: key for (key,value) in d.items()}

def process_file(filename):
    atoms = []
    predicates = set()
    constants = set()
    with open(filename) as f:
        data = f.read().replace('\n', '')
        result = prolog_sentences.parseString(data)
        for idx in range(len(result['facts'])):
            fact = result['facts'][idx]
            predicate = result['relationship'][idx]
            terms = [Term(False, term) for term in result['arguments'][idx]]
            term_var = [Term(True, f'X_{i}') for i in range(len(terms))]

            predicates.add(Atom(term_var, predicate))
            atoms.append(Atom(terms, predicate))
            constants.update([Term(False, term) for term in result['arguments'][idx]])
    return atoms, predicates, constants

def main(task, epochs : int = 100, steps : int = 1, cuda : bool = False, inv : int = 10,
        debug : bool = False, norm : str = 'max', norm_weight : float = 0.0):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    dilp.set_norm(norm)

    dev = torch.device(0) if cuda else torch.device('cpu')


    if  inv<0:
            raise Exception('The number of invented predicates must be >= 0')


    B, pred_f, constants_f = process_file('%s/facts.dilp' % task)
    P, target_p, constants_p = process_file('%s/positive.dilp' % task)
    N, target_n, constants_n = process_file('%s/negative.dilp' % task)

    if not (target_p == target_n):
        raise Exception('Positive and Negative files have different targets')
    elif not len(target_p) == 1:
        raise Exception('Can learn only one predicate at a time')
    elif not constants_n.issubset(constants_f) or not constants_p.issubset(constants_f):
        raise Exception(
            'Constants not in fact file exists in positive/negative file')

    invented_names= ["inv_"+str(x) for x in range(inv)]
    target = list(target_p)[0].predicate
    target_arity = P[0].arity
    count_examples = len(P)+len(N)
    pred_dim = len(pred_f)+inv+2
    atom_dim = len(constants_f)
    rules_dim = (pred_dim**2)*81
    fact_names = [ x.predicate for x in pred_f]
    true_facts = B+P
    target_facts = P+N
    var_names = [Term(True, f'X_{i}') for i in range(3)]

    pred_dict : Dict[int, str] = dict(zip(list(range(pred_dim)),[ x.predicate for x in pred_f]+ [target,"false"]+invented_names))
    atom_dict = dict(zip(list(range(atom_dim)),list(constants_f)))
    rules_dict = dict(zip(list(range(rules_dim)),[(x,y,z,w) for x in pred_dict.values() \
    for y in pred_dict.values() for z in range(9) for w in range(9)]))
    var_dict   = dict(zip(list(range(9)),[ (x,y) for x in var_names for y in var_names]))

    # reverse version of the dictionaries
    pred_dict_rev,atom_dict_rev,rules_dict_rev,var_dict_rev = revDict(pred_dict),revDict(atom_dict), \
    revDict(rules_dict),revDict(var_dict)

    base_val = torch.zeros([pred_dim, atom_dim, atom_dim], dtype=torch.float, device=dev)
    body_predicates = torch.zeros([pred_dim, 2, rules_dim,2], dtype=torch.long, device=dev)
    variable_choices = torch.zeros([pred_dim, 2, rules_dim,2], dtype=torch.long, device=dev)
    targets = torch.full([count_examples,target_arity+1], pred_dict_rev[target], dtype=torch.long, device=dev)
    target_values = torch.zeros([count_examples], dtype=torch.float, device=dev)

    for x in range(pred_dim):
        for y in range(atom_dim):
            for z in range(atom_dim):
                if Atom([atom_dict[y],atom_dict[z]],pred_dict[x]) in true_facts:
                    base_val[x][y][z] = 1

    for x in range(pred_dim):
        for y in range(2):
            for z in range(rules_dim):
                for w in range(2):
                    body_predicates[x][y][z][w] = pred_dict_rev["false"] if pred_dict[x] in \
                    fact_names else pred_dict_rev[rules_dict[z][w]]
                    variable_choices[x][y][z][w] = int(rules_dict[z][2+w])

    for x in range(len(target_facts)):
        for y in range(target_arity):
            targets[x][y+1] = atom_dict_rev[target_facts[x].terms[y]]
        if x < len(P):
            target_values[x] = 1

    rulebook = dilp.Rulebook(body_predicates,variable_choices)
    logging.debug(f"{rulebook.body_predicates.shape=},{rulebook.variable_choices.shape=}")

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    weights : torch.nn.Parameter = torch.nn.Parameter(torch.rand(size=(pred_dim,2,rules_dim), device=dev))

    opt = torch.optim.RMSprop([weights], lr=1e-2)
    #opt = torch.optim.SGD([weights], lr=1e-2)
    print(f"done {rulebook.body_predicates.shape=} {rulebook.variable_choices.shape=}")
    opt = torch.optim.RMSprop([weights], lr=0.05)
    for epoch in tqdm(range(0, epochs)):
        opt.zero_grad()
        mse_loss = dilp.loss(base_val, rulebook=rulebook, weights = weights, targets=targets, target_values=target_values, steps=steps)
        mse_loss.backward()
        # with torch.no_grad():
        #     if weights.grad is not None:
        #         print(f"{weights.grad[2]}")
        #     #weights.grad = weights.grad / torch.max(weights.grad.norm(2, dim=-1, keepdim=True), torch.as_tensor(1e-8))
        #     pass
        opt.step()
        print(f"loss: {mse_loss.item()}")
        #print(f"{weights[2]=}")

    dilp.print_program(rulebook, weights, pred_dict)

def norm_loss(weights : torch.Tensor) -> torch.Tensor:
    x = weights.softmax(-1)
    x = x * (1-x)
    return x.sum()

if __name__ == "__main__":
    fire.Fire(main)
