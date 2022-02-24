import fire #type: ignore
import dilp
import torch
from tqdm import tqdm #type: ignore
import pyparsing as pp #type: ignore
import torch
import logging
#from core import Term, Atom
from typing import *
import itertools

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

def process_file(filename) -> Tuple[List[Tuple[str,List[str]]],Set[str],Set[str]]:
    atoms : List[Tuple[str,List[str]]] = []
    predicates : Set[str] = set()
    constants : Set[str] = set()
    with open(filename) as f:
        data = f.read().replace('\n', '')
        result = prolog_sentences.parseString(data)
        for idx in range(len(result['facts'])):
            fact = result['facts'][idx]
            predicate = result['relationship'][idx]
            terms = result['arguments'][idx]

            predicates.add(predicate)
            atoms.append((predicate, terms))
            constants.update((term for term in result['arguments'][idx]))
    return atoms, predicates, constants

def main(task, epochs : int = 100, steps : int = 1, cuda : bool = False, inv : int = 10,
        debug : bool = False, norm : str = 'max', norm_weight : float = 1.0,
        optim : str = 'adam', lr : float = 0.05, clip : Optional[float] = None,
        unary : Set[str] = set(), init_rand : float = 10,
        layers : Optional[List[int]] = None,
        recursion : bool = False,
        seed : Optional[int] = 0, dropout : float = 0):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) #type: ignore

    dilp.set_norm(norm)

    dev = torch.device(0) if cuda else torch.device('cpu')


    if  inv<0:
            raise Exception('The number of invented predicates must be >= 0')


    true_facts, pred_f, constants_f = process_file('%s/facts.dilp' % task)
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
    target = next(iter(target_p))
    target_arity = len(P[0][1])
    count_examples = len(P)+len(N)
    pred_dim = len(pred_f)+inv+1
    atom_dim = len(constants_f)
    rules_dim = ((pred_dim-1)**2)*81
    #fact_names = [ x.predicate for x in pred_f]
    target_facts = P+N
    #var_names = [Term(True, f'X_{i}') for i in range(3)]

    pred_dict : Dict[int, str] = dict(zip(list(range(pred_dim)),[target]+list(pred_f)+invented_names))
    atom_dict = dict(zip(list(range(atom_dim)),list(constants_f)))
    #rules_dict = dict(zip(list(range(rules_dim)),[(x,y,z,w) for x in pred_dict.values() \
    #        for y in pred_dict.values() for z in range(9) for w in range(9)]))
    #var_dict   = dict(zip(list(range(9)),[ (x,y) for x in var_names for y in var_names]))

    print(f"{list(pred_dict.items())=}")

    # reverse version of the dictionaries
    pred_dict_rev,atom_dict_rev = revDict(pred_dict),revDict(atom_dict)
    unary_preds : Set[int] = set(pred_dict_rev[u] for u in unary)
    invented_preds = set(pred_dict_rev[f"inv_{i}"] for i in range(0, inv))

    if layers is None:
        layer_dict : Dict[int, int] = dict()
    else:
        layer_dict = dict(zip(invented_preds, sum(([i for _ in range(layer)] for i, layer in enumerate(layers)), start=[])))

    base_val = torch.zeros([pred_dim, atom_dim, atom_dim], dtype=torch.float, device=dev)
    body_predicates = []
    variable_choices = []
    targets = torch.full([count_examples,target_arity+1], pred_dict_rev[target], dtype=torch.long, device=dev)
    target_values = torch.zeros([count_examples], dtype=torch.float, device=dev)

    print(f"{invented_preds=} {layer_dict=}")

    for atom in true_facts:
        pred_id = pred_dict_rev[atom[0]]
        atom_ids = [atom_dict_rev[x] for x in atom[1]]
        logging.debug(f"{atom[0]}({atom[1]}) -> {pred_id=} {atom_ids=}")
        base_val[pred_id][atom_ids[0]][atom_ids[1]] = 1

    for head_pred in range(pred_dim):
        ret_bp : List[Tuple[int,int]]= []
        ret_vc : List[Tuple[int,int]] = []
        head_name = pred_dict[head_pred]
        if head_name not in pred_f:
            for p1 in range(pred_dim):
                for p2 in range(pred_dim):
                    for v1, v2, v3, v4 in itertools.product(range(3),range(3),range(3),range(3)):
                        if p1 == head_pred and v1 == 0 and v2 == 1: continue #self recursion
                        if p2 == head_pred and v3 == 0 and v4 == 1: continue #self recursion
                        
                        if head_pred in unary_preds and 1 in {v1,v2,v3,v4}: continue #using second arg of unary target
                        
                        if p1 in unary_preds: v1 = v2
                        if p2 in unary_preds: v3 = v4
                        
                        if any(head_pred in invented_preds and p in invented_preds and p < head_pred for p in {p1,p2}): continue

                        if not recursion and head_pred in {p1, p2}: continue

                        #if head_pred == 4 and p1 == 6:
                            #for p in {p1,p2}:
                                #print(f"{layers is not None=} {head_pred in invented_preds=} {p != head_pred=} {p in invented_preds=} {layer_dict[head_pred]+1 != layer_dict[p]=}")
                        if any(layers is not None and head_pred in invented_preds and p != head_pred and p in invented_preds and layer_dict[head_pred]+1 != layer_dict[p] for p in {p1, p2}): continue

                        if any(layers is not None and head_pred == 0 and p in invented_preds and layer_dict[p] != 0 for p in {p1,p2}): continue #main pred only calls first layer

                        #if head_pred in invented_preds and p1 != head_pred and p1 in invented_preds: continue #flat inveted only
                        #if head_pred in invented_preds and p2 != head_pred and p2 in invented_preds: continue

                        vc1 = v1 * 3 + v2
                        vc2 = v3 * 3 + v4
                        ret_bp.append((p1,p2))
                        ret_vc.append((vc1,vc2))
                        #logging.info(f"rule {pred_dict[head_pred]}(0,1) :- {pred_dict[p1]}({v1},{v2}), {pred_dict[p2]}({v3,v4})")

        bp = torch.as_tensor(ret_bp, device=dev, dtype=torch.long).unsqueeze(0).repeat(2,1,1)
        vc = torch.as_tensor(ret_vc, device=dev, dtype=torch.long).unsqueeze(0).repeat(2,1,1)
        body_predicates.append(bp)
        variable_choices.append(vc)
        del bp, vc, ret_bp, ret_vc

    for x in range(len(target_facts)):
        for y in range(target_arity):
            targets[x][y+1] = atom_dict_rev[target_facts[x][1][y]]
            target_values[x] = float(x < len(P))

    rulebook = dilp.Rulebook(body_predicates,variable_choices)

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    weights : List[torch.nn.Parameter] = [
        torch.nn.Parameter(torch.rand(size=bp.shape[:-1], device=dev)*init_rand)
            for bp in rulebook.body_predicates]

    #opt = torch.optim.SGD([weights], lr=1e-2)
    print(f"done")
    if optim == 'rmsprop':
        opt : torch.optim.Optimizer = torch.optim.RMSprop(weights, lr=lr)
    elif optim == 'adam':
        opt = torch.optim.Adam(weights, lr=lr)
    else:
        assert False

    for epoch in tqdm(range(0, int(epochs))):
        opt.zero_grad()

        if dropout != 0:
            moved : Sequence[torch.Tensor] = [w + torch.rand_like(w) * dropout for w in weights]
        else:
            moved = weights
        mse_loss, _ = dilp.loss(base_val, rulebook=rulebook, weights = moved, targets=targets, target_values=target_values, steps=steps)
        mse_loss = mse_loss.sum()
        mse_loss.backward()

        n_loss : torch.Tensor = sum((norm_loss(w) for w in weights), start=torch.zeros(size=(), device=dev)) \
                 * norm_weight / sum(w.numel() for w in weights)
        n_loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(weights, clip)

        opt.step()

        print(f"mse loss: {mse_loss.item()} norm loss: {n_loss.item()}")

    dilp.print_program(rulebook, weights, pred_dict)

    _, report = dilp.loss(base_val, rulebook=rulebook, weights = weights, targets=targets, target_values=target_values, steps=steps)
    print('weighted report:\n', report.detach().cpu().numpy())
    c0 = (torch.nn.functional.one_hot(w.max(-1)[1], w.shape[-1]) for w in weights)
    crisp : Sequence[torch.Tensor] = [c.float().where(c == 1, torch.as_tensor(-float('inf'), device=c.device)) for c in c0]
    _, crisp_report = dilp.loss(base_val, rulebook=rulebook, weights = crisp, targets=targets, target_values=target_values, steps=steps)
    print('crisp report:\n', crisp_report.detach().cpu().numpy())

def norm_loss(weights : torch.Tensor) -> torch.Tensor:
    #x = weights.softmax(-1)
    x = weights
    #x = x * (1-x)
    logsoftmax = x.log_softmax(-1)
    softmax = logsoftmax.exp()
    x = (softmax * logsoftmax)
    return -x.sum()

if __name__ == "__main__":
    fire.Fire(main)
