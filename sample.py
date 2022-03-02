import fire #type: ignore
import dilp
import torch
from tqdm import tqdm #type: ignore
import pyparsing as pp #type: ignore
import torch
import logging
#from core import Term, Atom
from typing import *
import GPUtil #type: ignore
import itertools
import numpy
import random

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

def merge_pad(ts : List[torch.Tensor], dim : int = 0, newdim : int = 0) -> torch.Tensor:
    max_len = max(len(t) for t in ts)
    def required_shape(t : torch.Tensor) -> List[int]:
        ret = list(t.shape)
        ret[dim] = max_len - ret[dim]
        return ret
    ts = list(torch.cat((t, torch.zeros(size=required_shape(t), device=t.device, dtype=t.dtype)), dim=dim).unsqueeze(newdim) for t in ts)
    return torch.cat(ts, dim=newdim)

def merge_mask(ts : List[torch.Tensor], dim : int = 0, newdim : int = 0) -> torch.Tensor:
    max_len = max(len(t) for t in ts)
    def required_shape(t : torch.Tensor) -> List[int]:
        ret = list(t.shape)
        ret[dim] = max_len - ret[dim]
        return ret
    ts = list(torch.cat((torch.ones_like(t), 
        torch.as_tensor(-float('inf'), device=t.device).repeat(required_shape(t))), dim=dim).unsqueeze(newdim) for t in ts)
    return torch.cat(list(ts), dim=newdim)

def mask(t : torch.Tensor, rulebook : dilp.Rulebook) -> torch.Tensor:
    return t.where(rulebook.mask.unsqueeze(1).unsqueeze(1), torch.zeros(size=(),device=t.device))

def masked_softmax(t : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
    t = t.where(mask, torch.as_tensor(-float('inf'), device=t.device)).softmax(-1)
    t = t.where(t.isnan().logical_not(), torch.as_tensor(0.0, device=t.device)) #type: ignore
    return t

def main(task, epochs : int = 100, steps : int = 1, cuda : Optional[int] = None, inv : int = 0,
        debug : bool = False, norm : str = 'max', norm_weight : float = 1.0,
        optim : str = 'adam', lr : float = 0.05, clip : Optional[float] = None,
        unary : Set[str] = set(), init_rand : float = 10,
        layers : Optional[List[int]] = None, info : bool = False,
        recursion : bool = True, normalize_threshold : Optional[float] = None,
        invented_recursion : bool = False, batch_size : Optional[int] = None,
        normalize_gradients : Optional[float] = None,
        init : str = 'uniform',
        entropy_weight_step = 1e-2,
        seed : Optional[int] = None, dropout : float = 0):
    if info:
        logging.getLogger().setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if seed is not None:
        torch.use_deterministic_algorithms(True) #type: ignore
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) #type: ignore

    dilp.set_norm(norm)

    dev = torch.device(0) if cuda is not None else torch.device('cpu')


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
    #fact_names = [ x.predicate for x in pred_f]
    target_facts = positive_targets+negative_targets
    #var_names = [Term(True, f'X2700 kch to eur_{i}') for i in range(3)]

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
    body_predicates : List[torch.Tensor] = []
    variable_choices : List[torch.Tensor] = []
    targets = torch.full([count_examples,target_arity+1], pred_dict_rev[target], dtype=torch.long, device=dev)
    target_values = torch.zeros([count_examples], dtype=torch.float, device=dev)

    print(f"{invented_preds=} {layer_dict=}")

    for atom in true_facts:
        pred_id = pred_dict_rev[atom[0]]
        atom_ids = [atom_dict_rev[x] for x in atom[1]]
        logging.info(f"{atom[0]}({atom[1]}) -> {pred_id=} {atom_ids=}")
        base_val[pred_id][atom_ids[0]][atom_ids[1]] = 1

    for head_pred in range(pred_dim):
        ret_bp : List[int]= []
        ret_vc : List[int] = []
        head_name = pred_dict[head_pred]
        if head_name not in pred_f:
            for p in range(pred_dim):
                #for p2 in range(pred_dim):
                    for a, b in itertools.product(range(3),range(3)):
                        if p == head_pred and a == 0 and b == 1: continue #self recursion
                        
                        if head_pred in unary_preds and 1 in {a,b}: continue #using second arg of unary target
                        
                        if p in unary_preds: a == b
                        
                        #if any(head_pred in invented_preds and p in invented_preds and p < head_pred for p in {p1,p2}): continue

                        if not recursion and head_pred == p: continue

                        if not invented_recursion and head_pred in invented_preds and p in {head_pred, 0}: continue

                        if head_pred != 0 and p == 0: continue

                        if layers is not None and head_pred in invented_preds and p != head_pred and p in invented_preds and layer_dict[head_pred]+1 != layer_dict[p]: continue

                        if layers is not None and head_pred == 0 and p in invented_preds and layer_dict[p] != 0: continue #main pred only calls first layer

                        vc1 = a * 3 + b
                        ret_bp.append(p)
                        ret_vc.append(vc1)
                        #logging.info(f"rule {pred_dict[head_pred]}(0,1) :- {pred_dict[p1]}({v1},{v2}), {pred_dict[p2]}({v3,v4})")

        bp = torch.as_tensor(ret_bp, device=dev, dtype=torch.long)
        vc = torch.as_tensor(ret_vc, device=dev, dtype=torch.long)
        body_predicates.append(bp)
        variable_choices.append(vc)
        logging.info(f"predicate {head_name} ({head_pred}) rules {bp.shape} {vc.shape}")
        del bp, vc, ret_bp, ret_vc

    for x in range(len(target_facts)):
        for y in range(target_arity):
            targets[x][y+1] = atom_dict_rev[target_facts[x][1][y]]
            target_values[x] = float(x < len(positive_targets))

    rulebook = dilp.Rulebook(merge_pad(body_predicates),merge_pad(variable_choices),merge_pad([torch.ones_like(t).bool() for t in body_predicates]))

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    shape = rulebook.body_predicates.shape
    #weights : torch.nn.Parameter = 
    weights : torch.nn.Parameter = torch.nn.Parameter(torch.normal(mean=torch.zeros(size=[shape[0],2,2,shape[1]], device=dev), std=init_rand)) \
        if init == 'normal' else torch.nn.Parameter(torch.rand([shape[0],2,2,shape[1]], device=dev) * init_rand)
        #torch.nn.Parameter(torch.rand(size=(2, 2, len(bp)), device=dev)*init_rand)
        #torch.nn.Parameter(torch.normal(torch.zeros(size=bp.shape[:-1], device=dev), init_rand))
        #    for bp in rulebook.body_predicates]
    #adjust_weights(weights)
    logging.info(f"{weights.shape=}")

    #opt = torch.optim.SGD([weights], lr=1e-2)
    print(f"done")
    if optim == 'rmsprop':
        opt : torch.optim.Optimizer = torch.optim.RMSprop([weights], lr=lr)
    elif optim == 'adam':
        opt = torch.optim.Adam([weights], lr=lr)
    elif optim == 'sgd':
        opt = torch.optim.SGD([weights], lr=lr)
    else:
        assert False
        
    entropy_enabled = normalize_threshold is None
    entropy_weight = 0.0

    for epoch in (tq := tqdm(range(0, int(epochs)))):
        opt.zero_grad()

        if dropout != 0:
            moved : torch.Tensor = weights.softmax(-1) * (1-dropout) * torch.rand(weights.shape, device=weights.device)
        else:
            moved = masked_softmax(weights, rulebook.mask.unsqueeze(1).unsqueeze(1))
        target_loss, _ = dilp.loss(base_val, rulebook=rulebook, weights = moved, targets=targets, target_values=target_values, steps=steps)
        report_loss = target_loss.mean()
        if batch_size is not None:
            assert batch_size <= len(positive_targets) and batch_size <= len(negative_targets)
            chosen_positive = torch.randperm(len(positive_targets), device=dev) >= len(positive_targets) - batch_size
            chosen_negative = torch.randperm(len(negative_targets), device=dev) >= len(negative_targets) - batch_size
            chosen = torch.cat((chosen_positive, chosen_negative))
            target_loss = target_loss.where(chosen, torch.as_tensor(0.0, device=dev)).sum() / chosen.sum()
        else:
            target_loss = target_loss.mean()
        target_loss.backward()
            
        if normalize_threshold is not None and report_loss.item() < normalize_threshold:
            entropy_enabled = True
            
        if entropy_enabled:
            entropy_loss : torch.Tensor = norm_loss(mask(weights, rulebook))
            entropy_loss = mask(entropy_loss, rulebook).mean()
            if entropy_weight < 1.0 and normalize_threshold is not None and report_loss.item() < normalize_threshold:
                entropy_weight += entropy_weight_step
            entropy_loss *= entropy_weight * norm_weight
            entropy_loss.backward()
        else:
            entropy_loss = torch.as_tensor(0.0)

        if clip is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(weights, clip)

        if normalize_gradients is not None:
            with torch.no_grad():
                for w in weights:
                    w /= w.sum(-1)
                    w *= normalize_gradients

        opt.step()
        #adjust_weights(weights)

        tq.set_postfix(target_loss = report_loss.item(), entropy_loss = entropy_loss.item(), batch_loss = target_loss.item(), entropy_weight=entropy_weight * norm_weight)

        logging.info(f"target loss: {report_loss.item()} entropy loss: {entropy_loss.item()}")

    dilp.print_program(rulebook, mask(weights, rulebook), pred_dict)

    final_loss, fuzzy_report = dilp.loss(base_val, rulebook=rulebook, weights = masked_softmax(weights, rulebook.mask.unsqueeze(1).unsqueeze(1)), targets=targets, target_values=target_values, steps=steps)
    crisp = mask(torch.nn.functional.one_hot(weights.max(-1)[1], weights.shape[-1]).float(), rulebook)
    #crisp : Sequence[torch.Tensor] = [c.float().where(c == 1, torch.as_tensor(-float('inf'), device=c.device)) for c in c0]
    crisp_loss, crisp_report = dilp.loss(base_val, rulebook=rulebook, weights = crisp, targets=targets, target_values=target_values, steps=steps)
    report = torch.cat([target_values.unsqueeze(1), fuzzy_report.unsqueeze(1), crisp_report.unsqueeze(1)], dim=1).detach().cpu().numpy()
    print(f'final_loss: {final_loss.mean().item():.5f} crisp_loss: {crisp_loss.mean().item():.5f}:\n', report)

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
