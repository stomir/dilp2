from sys import exec_prefix
import torch
import logging
import math
import loader
import itertools
from typing import *
from loader import Problem
from torcher import WorldsBatch, TargetSet, TargetType, Rulebook

import weird

def disjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a + b - a * b
def disjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    if a.shape[dim] == 2:
        disjunction2_prod(*a.split(1, dim=dim)).squeeze(dim)
    return 1 - ((1 - a).prod(dim))
def conjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a * b
def conjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.prod(dim=dim)

def disjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.max(a, b)
def disjunction2_max_left(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a.where(a>=b, b)

def disjunction2_max_even(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.max(a, b)
def disjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.max(a, dim=dim)[0]
def conjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.min(a, b)
    #return a.where(a<b, b)
def conjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.min(dim=dim)[0]

def conjunction_dim_yager(p : float) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def yager(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
        return torch.min(torch.as_tensor(1.0, device=a.device), a.pow(p).sum(dim).pow(1/p))
    return yager

def generalized_mean(p : float) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def gm(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
        return a.pow(p).mean(dim=dim).pow(1/p)
    return gm
    

conjunction_body_pred : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = conjunction2_max
disjunction_quantifier : Callable[[torch.Tensor, int], torch.Tensor] = disjunction_dim_max
disjunction_steps : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  = disjunction2_max
disjunction_clauses : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = disjunction2_max

def soft_max(t : torch.Tensor, d : int, p : float) -> torch.Tensor:
    return (t * (t * p).softmax(dim=d)).sum(dim=d)

def dim2two(f : Callable[[torch.Tensor, int], torch.Tensor]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return lambda a,b: f(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), 0)

def set_norm(norm_name : str, p : float):
    global conjunction_body_pred, disjunction_quantifier, disjunction_steps, disjunction_clauses
    if norm_name == 'max':
        conjunction_body_pred = conjunction2_max
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_max
        disjunction_clauses = disjunction2_max
    elif norm_name == 'prod':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = disjunction_dim_prod
        disjunction_steps = disjunction2_prod
        disjunction_clauses = disjunction2_prod
    elif norm_name == 'mixed':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_max
        disjunction_clauses = disjunction2_max
    elif norm_name == 'soft_max':
        conjunction_body_pred = dim2two(lambda t,d: soft_max(t,d,-p))
        disjunction_quantifier = lambda t,d: soft_max(t,d,p)
        disjunction_steps = dim2two(lambda t,d: soft_max(t,d,p))
        disjunction_clauses = dim2two(lambda t,d: soft_max(t,d,p))
    elif norm_name == 'weird':
        conjunction_body_pred = weird.WeirdMin.apply #type: ignore
        disjunction_quantifier = weird.WeirdMaxDim.apply #type: ignore
        disjunction_steps = weird.WeirdMax.apply #type: ignore
        disjunction_clauses = weird.WeirdMaxDim.apply #type: ignore
    elif norm_name == 'weird2':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = weird.WeirdMax2Dim.apply
        disjunction_steps = disjunction2_max
        disjunction_clauses = disjunction2_max
    elif norm_name == 'weird22':
        conjunction_body_pred = lambda a,b: 1-(weird.WeirdMax2.apply(1-a,1-b))
        disjunction_quantifier = weird.WeirdMax2Dim.apply
        disjunction_steps = weird.WeirdMax2.apply
        disjunction_clauses = weird.WeirdMax2.apply
    elif norm_name == 'mixed_left':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_max_left
        disjunction_clauses = disjunction2_max
    elif norm_name == 'dilp':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_prod
        disjunction_clauses = disjunction2_max
    elif norm_name == 'dilpB':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_max_left
        disjunction_clauses = disjunction2_prod
    elif norm_name == 'dilpB2':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_max
        disjunction_clauses = disjunction2_prod
    elif norm_name == 'dilpC':
        conjunction_body_pred = conjunction2_max
        disjunction_quantifier = disjunction_dim_max
        disjunction_steps = disjunction2_max
        disjunction_clauses = disjunction2_prod
    elif norm_name == 'krieken':
        conjunction_body_pred = conjunction2_prod
        disjunction_quantifier = generalized_mean(p = p)
        disjunction_steps = disjunction2_max
        disjunction_clauses = disjunction2_max
    else:
        assert False, f"wrong norm name {norm_name=}"

def squeeze_into(v : torch.Tensor, dim : int, dim2 : int) -> torch.Tensor:
    shape = list(v.shape)
    shape[dim] *= shape[dim2]
    del shape[dim2]
    return v.reshape(shape)

def extend_val(val : torch.Tensor, vars : int = 3) -> torch.Tensor:
    ret = []
    shape = list(val.shape) + [val.shape[-1] for _ in range(0, vars - 2)]
    valt = val.transpose(2, 3)
    for i, (arg1, arg2) in enumerate(itertools.product(range(vars), range(vars))):
        v = val
        if arg1 == arg2:
            v = v.diagonal(dim1=2,dim2=3)
            for _ in range(0, arg1):
                v = v.unsqueeze(2)
            while len(v.shape) < vars+2:
                v = v.unsqueeze(-1)
        else:
            if arg2 < arg1:
                v = valt
            unused = (x for x in range(0, vars) if x not in {arg1, arg2})
            for u in unused:
                v = v.unsqueeze(u + 2)
        v = torch.broadcast_to(v, shape) #type: ignore
        v = v.unsqueeze(2)
        ret.append(v)
    return torch.cat(ret, dim=2)

def infer_single_step(ex_val : torch.Tensor, 
        rule_weights : torch.Tensor,
        split : int,
        ) -> torch.Tensor:

    rule_weights = rule_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #atoms
    rule_weights = rule_weights.unsqueeze(0) #worlds

    ex_val = squeeze_into(ex_val, 1, 2)

    if split == 2:
        logging.debug(f"{ex_val.shape=} {rule_weights.shape=}")
        
        ex_val = ex_val.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        ex_val = ex_val#.unsqueeze(-5).unsqueeze(-5)
        ex_val = ex_val * rule_weights 
        ex_val = ex_val.sum(dim = -4)
        #ex_val = ex_val.where(ex_val.isnan().logical_not(), torch.zeros(size=(), device=ex_val.device)) #type: ignore
        
        control_valve = torch.max(ex_val.detach()-1, torch.as_tensor(0.0, device=ex_val.device))
        ex_val = ex_val - control_valve

        #conjuction of body predictes
        ex_val = conjunction_body_pred(ex_val[:,:,:,0,:,:,:], ex_val[:,:,:,1,:,:,:])
        #existential quantification
        ex_val = disjunction_quantifier(ex_val, -1)
        #disjunction on clauses
        ex_val = disjunction_clauses(ex_val[:,:,0,:,:], ex_val[:,:,1,:,:])
        logging.debug(f"returning {ex_val.shape=}")
        return ex_val
    elif split == 1:
        ex_val = conjunction_body_pred(ex_val.unsqueeze(1), ex_val.unsqueeze(2))
        logging.debug(f"ex_val big {ex_val.shape=}")

        ex_val = squeeze_into(ex_val, 1, 2)
        ex_val = ex_val.unsqueeze(0).unsqueeze(0)
        ex_val = ex_val * rule_weights 
        ex_val = ex_val.sum(dim = -4)

        #existential quantification
        ex_val = disjunction_quantifier(ex_val, -1)
        #disjunction on clauses
        ex_val = disjunction_clauses(ex_val[:,:,0,:,:], ex_val[:,:,1,:,:])
        logging.debug(f"returning {ex_val.shape=}")
        return ex_val
    elif split == 0:
        ex_val = conjunction_body_pred(ex_val.unsqueeze(1), ex_val.unsqueeze(2))
        ex_val = squeeze_into(ex_val, 1, 2)

        #existential quantification
        ex_val = disjunction_quantifier(ex_val, -1)

        ex_val = ex_val.unsqueeze(0).unsqueeze(0)
        ex_val = ex_val * rule_weights 
        ex_val = ex_val.sum(dim = -4)

        
        ex_val = disjunction_quantifier(ex_val, -1)
        #disjunction on clauses

        ex_val = disjunction_clauses(ex_val.unsqueeze(2), ex_val.unsqueeze(3))
        ex_val = squeeze_into(ex_val, 2, 3)

        ex_val = ex_val.unsqueeze(0)
        ex_val = ex_val * rule_weights 
        ex_val = ex_val.sum(dim = -3)

        logging.debug(f"returning {ex_val.shape=}")
        return ex_val
    else:
        raise NotImplementedError(f"{split=}")


def infer_steps_on_devs(steps : int, base_val : torch.Tensor,
        return_dev : torch.device, devices : Sequence[torch.device],
        rule_weights : torch.Tensor,
        split : int,
        ) -> torch.Tensor:
    pred_count : int = rule_weights.shape[0]
    per_dev = math.ceil(pred_count / len(devices))
    
    rule_weights_ : List[torch.Tensor] = []
    for i, dev in enumerate(devices):
        rule_weights_.append(rule_weights[i*per_dev:(i+1)*per_dev].to(dev, non_blocking=True))

    val = base_val
    for step in range(steps):
        rets = []
        for i, dev in enumerate(devices):
            rets.append(infer_single_step(
                ex_val = extend_val(val.to(dev, non_blocking=True)),
                split=split,
                rule_weights = rule_weights_[i]))
        val = disjunction_steps(val, torch.cat([t.to(return_dev, non_blocking=True) for t in rets], dim=1))
    
    return val



def infer_steps(steps : int, base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor, 
            problem : Problem,
            split : int,
            vars : int = 3) -> torch.Tensor:
    val = base_val
    #non-bk weights

    weights = weights[len(problem.bk):]
    bk_zeros = torch.zeros([val.shape[0], len(problem.bk), val.shape[2], val.shape[3]], device = val.device)

    #vals : List[torch.Tensor] = []
    for i in range(0, steps):
        val2 = extend_val(val, vars)
        val2 = infer_single_step(ex_val = val2, \
            split=split,
            rule_weights = weights)
        val2 = torch.cat([bk_zeros, val2], dim=1)
        
        #weird grad norm
        #val2 = weird.WeirdNorm.apply(val2, (-1,-2,-3), 1.0)
        
        assert val.shape == val2.shape, f"{i=} {val.shape=} {val2.shape=}"
        #vals.append(val2.unsqueeze(0))
        val = disjunction_steps(val, val2)
    return val

def infer(base_val : torch.Tensor,
            rulebook : Rulebook,
            weights : torch.Tensor,
            problem : Problem,
            steps : int,
            split : int,
            devices : Optional[Sequence[torch.device]] = None,
            ) -> torch.Tensor:
    if devices is None:
        return infer_steps(steps, base_val, rulebook, weights, problem, vars = 3, split=split)
    else:
        return infer_steps_on_devs(steps, base_val, weights.device, devices, split = split,
            rule_weights=weights)

def loss_value(values : torch.Tensor, target_type : loader.TargetType, reduce : bool = True) -> torch.Tensor:
    if len(values) == 0:
        return torch.as_tensor(0.0, device=values.device)
    if target_type == loader.TargetType.POSITIVE:
        ret = -(values + 1e-10).log()
        if reduce:
            ret = ret.mean()
        return ret
    else:
        return loss_value(1-values, target_type=loader.TargetType.POSITIVE, reduce=reduce)

base_filter = filter
def extract_targets(vals : torch.Tensor, targets : torch.Tensor, filter : Optional[Callable[[Tuple[int,int,int]], bool]] = None) -> torch.Tensor:
    if filter is not None:
        targets = list(base_filter(filter, targets)) #type: ignore
    if len(targets) == 0:
        return torch.as_tensor([], device=targets.device)
    return vals[targets[:,0],targets[:,1],targets[:,2],targets[:,3]]

def loss(vals : torch.Tensor, batch : WorldsBatch, filter : Optional[Callable[[Tuple[int,int,int]], bool]] = None) -> torch.Tensor:
    pos = extract_targets(vals, batch.positive_targets.idxs, filter = filter)
    neg = extract_targets(vals, batch.negative_targets.idxs, filter = filter)
    pos = loss_value(values = pos, target_type = TargetType.POSITIVE, reduce=True)
    neg = loss_value(values = neg, target_type = TargetType.NEGATIVE, reduce=True)
    return (pos + neg) / 2

def var_choices(n : int, vars : int = 3) -> List[int]:
    return [int(n) // vars, n % vars]

def body_pred_str(choice : int, problem : loader.Problem) -> str:
    chosen_pred = choice // 9
    chosen_vs = choice % 9
    vs = ','.join(map(lambda v: chr(ord('A')+v), var_choices(chosen_vs)))
    return f'{problem.predicate_name[chosen_pred]}({vs})'

def clause_str(pred_name : str, choice1 : int, choice2 : int, problem : loader.Problem) -> str:
    return f"{pred_name.rjust(10, ' ')}(A,B) :- {body_pred_str(choice1, problem)}, {body_pred_str(choice2, problem)}."

def print_program(problem : loader.Problem, weights : torch.Tensor, split : int):
    pred_dim = len(problem.predicate_name)
    if split == 2:
        for pred in list(problem.targets) + list(problem.invented):
            pred_name = problem.predicate_name[pred]
            for clause in range(2):
                choices = list(weights[pred,clause].max(-1).indices.cpu())
                print(clause_str(pred_name, choices[0], choices[1], problem))
    elif split == 1:
        for pred in list(problem.targets) + list(problem.invented):
            pred_name = problem.predicate_name[pred]
            d = pred_dim * 9
            for clause in range(2):
                
                rule_no = weights[pred,clause].max(-1).values.item()
                print(clause_str(pred_name, rule_no // d, rule_no % d, problem))
    elif split == 0:
        for pred in list(problem.targets) + list(problem.invented):
            pred_name = problem.predicate_name[pred]
            d = pred_dim * 9
            d2 = d * d
            rule_no = weights[pred].max(-1).values.item()
            for clause_choice in (rule_no // d2, rule_no % d2):
                print(clause_str(pred_name, clause_choice // d, clause_choice % d, problem))
    else: 
        raise NotImplementedError
