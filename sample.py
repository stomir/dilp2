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
def checkFunctional(s, functional):
    ss = s.replace('\n', '').split("::")
    print(s,ss,functional)
    if (functional and len(ss) != 4):
        raise Exception('Incorrect Mode Declaration!')
    if (not functional and len(ss) != 3):
        raise Exception('Incorrect Mode Declaration!')
    return ss
def process_mode_file(filename,functional):
    with open(filename) as f:
        data = [ checkFunctional(x,functional) for x in f]
        results = [[] for x in range(3)]
        for d in data:
            for i in range(1,len(d)):
                if d[i]=='+':
                    results[i-1].append(d[0])
        return *tuple(results),len(data)
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

def main(task, epochs : int = 100, steps : int = 1, cuda : bool = False, inv : int = 0,
        debug : bool = False, norm : str = 'max', norm_weight : float = 1.0,
        optim : str = 'adam', lr : float = 0.05, clip : Optional[float] = None,
        unary : Set[str] = set(), init_rand : float = 10,
        layers : Optional[List[int]] = None, info : bool = False,
        recursion : bool = True, normalize_threshold : Optional[float] = None,
        seed : Optional[int] = 0, dropout : float = 0,
        mode : bool = False, functional : bool = False):
    if info:
        logging.getLogger().setLevel(logging.INFO)
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
    positive_targets, target_p, constants_p = process_file('%s/positive.dilp' % task)
    negative_targets, target_n, constants_n = process_file('%s/negative.dilp' % task)

    if not (target_p == target_n):
        raise Exception('Positive and Negative files have different targets')
    elif not len(target_p) == 1:
        raise Exception('Can learn only one predicate at a time')
    elif not constants_n.issubset(constants_f) or not constants_p.issubset(constants_f):
        raise Exception('Constants not in fact file exists in positive/negative file')
        #pass

    invented_names= ["inv_"+str(x) for x in range(inv)]
    target = next(iter(target_p))
    target_arity = len(positive_targets[0][1])
    count_examples = len(positive_targets)+len(negative_targets)
    pred_dim = len(pred_f)+inv+1
    atom_dim = len(constants_f)
    rules_dim = ((pred_dim-1)**2)*81
    inPosOne, inPosTwo,func,modeCount = process_mode_file('%s/mode.dilp' % task,functional) if mode else ([],[],[],pred_dim)
    if modeCount!= pred_dim:
            raise Exception('Insufficient or too many Mode Declarations!')

    #fact_names = [ x.predicate for x in pred_f]
    target_facts = positive_targets+negative_targets
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

    #mode Declaration stuff
    inPosOne = [pred_dict_rev[x] for x in inPosOne]
    inPosTwo = [pred_dict_rev[x] for x in inPosTwo]
    func = [pred_dict_rev[x] for x in func]

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
        logging.info(f"{atom[0]}({atom[1]}) -> {pred_id=} {atom_ids=}")
        base_val[pred_id][atom_ids[0]][atom_ids[1]] = 1

    for head_pred in range(pred_dim):
        ret_bp : List[Tuple[int,int]]= []
        ret_vc : List[Tuple[int,int]] = []
        head_name = pred_dict[head_pred]
        if head_name not in pred_f:
            for p1 in range(pred_dim):
                for p2 in range(pred_dim):
                    for v1, v2, v3, v4 in itertools.product(range(3),range(3),range(3),range(3)):
                        if any(p == head_pred and a == 0 and b == 1 for (p,a,b) in {(p1,v1,v2),(p2,v3,v4)} ): continue #self recursion

                        if head_pred in unary_preds and 1 in {v1,v2,v3,v4}: continue #using second arg of unary target
                        if head_pred in unary_preds and not 0 in {v1,v2,v3,v4}: continue #datalog on unary target
                        if not head_pred in unary_preds and not ( 0 in {v1,v2,v3,v4} or  1 in {v1,v2,v3,v4}): continue #datalog on binary target

                        if p1 in unary_preds and v1!= v2: continue # only consider when vars are equal
                        if p2 in unary_preds and v3 != v4: continue # only consider when vars are equal

                        if p2 < p1 : continue # we should not check the symmetric case.

                        if functional and p1==p2 and p1 in func and v1 == v3 and v2 != v4: continue # should return the same thing for same arguments

                        # existential vars cannot be in in-positions without there being an out position in another predicate in the clause
                        if mode and p1 in inPosOne and  v1 == 2 and ((not 2 in {v3,v4}) or (v3 == 2 and  p2 in inPosOne) or (v4 == 2 and  p2 in inPosTwo)) : continue
                        if mode and p1 in inPosTwo and  v2 == 2 and ((not 2 in {v3,v4}) or (v3 == 2 and  p2 in inPosOne) or (v4 == 2 and  p2 in inPosTwo)) : continue
                        if mode and p2 in inPosOne and  v3 == 2 and ((not 2 in {v1,v2}) or (v1 == 2 and  p1 in inPosOne) or (v2 == 2 and  p1 in inPosTwo)) : continue
                        if mode and p2 in inPosTwo and  v4 == 2 and ((not 2 in {v1,v2}) or (v1 == 2 and  p1 in inPosOne) or (v2 == 2 and  p1 in inPosTwo)) : continue

                        # an out position of the head_predicate can never be an in position
                        if mode and not head_pred in inPosOne and  p1 in inPosOne and v1==0 : continue
                        if mode and not head_pred in inPosTwo and  p1 in inPosOne and v1==1 : continue
                        if mode and not head_pred in inPosOne and  p2 in inPosOne and v3==0 : continue
                        if mode and not head_pred in inPosTwo and  p2 in inPosOne and v3==1 : continue

                        #if any(head_pred in invented_preds and p in invented_preds and p < head_pred for p in {p1,p2}): continue

                        if not recursion and head_pred in {p1, p2}: continue

                        if any(layers is not None and head_pred in invented_preds and p != head_pred and p in invented_preds and layer_dict[head_pred]+1 != layer_dict[p] for p in {p1, p2}): continue

                        if any(layers is not None and head_pred == 0 and p in invented_preds and layer_dict[p] != 0 for p in {p1,p2}): continue #main pred only calls first layer

                        vc1 = v1 * 3 + v2
                        vc2 = v3 * 3 + v4
                        ret_bp.append((p1,p2))
                        ret_vc.append((vc1,vc2))
                        logging.debug(f"rule {pred_dict[head_pred]}(0,1) :- {pred_dict[p1]}({v1},{v2}), {pred_dict[p2]}({v3,v4})")

        bp = torch.as_tensor(ret_bp, device=dev, dtype=torch.long).unsqueeze(0).repeat(2,1,1)
        vc = torch.as_tensor(ret_vc, device=dev, dtype=torch.long).unsqueeze(0).repeat(2,1,1)
        body_predicates.append(bp)
        variable_choices.append(vc)
        logging.info(f"predicate {head_name} ({head_pred}) rules {bp.shape} {vc.shape}")
        del bp, vc, ret_bp, ret_vc

    for x in range(len(target_facts)):
        for y in range(target_arity):
            targets[x][y+1] = atom_dict_rev[target_facts[x][1][y]]
            target_values[x] = float(x < len(positive_targets))

    rulebook = dilp.Rulebook(body_predicates,variable_choices)

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    weights : List[torch.nn.Parameter] = [
        #torch.nn.Parameter(torch.normal(torch.zeros(size=bp.shape[:-1], device=dev), init_rand))
        torch.nn.Parameter(torch.rand(size=bp.shape[:-1], device=dev)*init_rand)
            for bp in rulebook.body_predicates]
    #adjust_weights(weights)
    logging.info(f"{weights[0].shape=}")

    #opt = torch.optim.SGD([weights], lr=1e-2)
    if optim == 'rmsprop':
        opt : torch.optim.Optimizer = torch.optim.RMSprop(weights, lr=lr)
    elif optim == 'adam':
        opt = torch.optim.Adam(weights, lr=lr)
    else:
        assert False

    for epoch in (tq := tqdm(range(0, int(epochs)))):
        opt.zero_grad()

        if dropout != 0:
            moved : Sequence[torch.Tensor] = [w.softmax(-1) * (1-dropout) +
                torch.distributions.Dirichlet(torch.ones_like(w)).sample() * dropout
                for w in weights]
        else:
            moved = [w.softmax(-1) for w in weights]
        target_loss, _ = dilp.loss(base_val, rulebook=rulebook, weights = moved, targets=targets, target_values=target_values, steps=steps)
        target_loss = target_loss.mean()
        target_loss.backward()

        if normalize_threshold is None or target_loss.item() < normalize_threshold:
            entropy_loss : torch.Tensor = sum((norm_loss(w) for w in weights), start=torch.zeros(size=(), device=dev)) \
                 * norm_weight / sum(w.numel() for w in weights)
            entropy_loss.backward()
        else:
            entropy_loss = torch.as_tensor(0.0)

        if clip is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(weights, clip)

        opt.step()
        #adjust_weights(weights)

        tq.set_postfix(target_loss = target_loss.item(), entropy_loss = entropy_loss.item())

        logging.info(f"target loss: {target_loss.item()} entropy loss: {entropy_loss.item()}")

    dilp.print_program(rulebook, weights, pred_dict)

    _, fuzzy_report = dilp.loss(base_val, rulebook=rulebook, weights = [w.softmax(-1) for w in weights], targets=targets, target_values=target_values, steps=steps)
    crisp = [torch.nn.functional.one_hot(w.max(-1)[1], w.shape[-1]) for w in weights]
    #crisp : Sequence[torch.Tensor] = [c.float().where(c == 1, torch.as_tensor(-float('inf'), device=c.device)) for c in c0]
    _, crisp_report = dilp.loss(base_val, rulebook=rulebook, weights = crisp, targets=targets, target_values=target_values, steps=steps)
    report = torch.cat([target_values.unsqueeze(1), fuzzy_report.unsqueeze(1), crisp_report.unsqueeze(1)], dim=1).detach().cpu().numpy()
    print('report:\n', report)

def adjust_weights(weights : List[torch.nn.Parameter]):
    with torch.no_grad():
            for w in weights:
                a = torch.max(torch.zeros(size=(), device=w.device), w)
                w[:] = a / a.sum(dim=1, keepdim=True)
                #assert (w.sum(-1) == 1).all(), f"{w.sum(-1)=}"

def norm_loss(weights : torch.Tensor) -> torch.Tensor:
    #x = weights.softmax(-1)
    x = weights
    #x = x * (1-x)
    logsoftmax = x.log_softmax(-1)
    softmax = logsoftmax.exp()
    x = (softmax * logsoftmax)
    #x = (x * x.log())
    return -x.sum()

if __name__ == "__main__":
    fire.Fire(main)
