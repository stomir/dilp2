from sys import exec_prefix
import torch
import logging
from typing import *

from zmq import device
import weird

class Rulebook(NamedTuple):
    body_predicates : torch.Tensor
    variable_choices : torch.Tensor
    mask : torch.Tensor

    def to(self, device : torch.device):
        return Rulebook(
            body_predicates=self.body_predicates.to(device),
            variable_choices=self.variable_choices.to(device),
            mask=self.mask.to(device)
        )

def disjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a + b - a * b
def disjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    if a.shape[dim] == 2:
        disjunction2_prod(*a.split(1, dim=dim))
    return 1 - ((1- a).prod(dim))
def conjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a * b
def conjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.prod(dim=dim)

def disjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    #return torch.max(a, b)
    return a.where(a>b, b)
def disjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.max(a, dim=dim)[0]
def conjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.min(a, b)
def conjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.min(dim=dim)[0]


disjunction2 = disjunction2_max
disjunction_dim = disjunction_dim_max
conjunction2 = conjunction2_max
conjunction_dim = conjunction_dim_max

def set_norm(norm_name : str):
    global disjunction2, disjunction_dim, conjunction2, conjunction_dim
    assert norm_name in {'max', 'prod', 'mixed', 'weird'}
    if norm_name == 'max':
        disjunction2 = disjunction2_max
        disjunction_dim = disjunction_dim_max
        conjunction2 = conjunction2_max
        conjunction_dim = conjunction_dim_max
    elif norm_name == 'prod':
        disjunction2 = disjunction2_prod
        disjunction_dim = disjunction_dim_prod
        conjunction2 = conjunction2_prod
        conjunction_dim = conjunction_dim_prod
    elif norm_name == 'mixed':
        disjunction2 = disjunction2_max
        disjunction_dim = disjunction_dim_max
        conjunction2 = conjunction2_prod
        conjunction_dim = conjunction_dim_prod
    elif norm_name == 'weird':
        disjunction2 = weird.WeirdMax.apply #type: ignore
        conjunction2 = weird.WeirdMin.apply #type: ignore
        disjunction_dim = weird.WeirdMaxDim.apply #type: ignore
        conjunction_dim = weird.WeirdMinDim.apply #type: ignore
    else:
        assert False

def extend_val(val : torch.Tensor, vars : int = 3) -> torch.Tensor:
    i = 0
    ret = []
    shape = list(val.shape) + [val.shape[-1] for _ in range(0, vars - 2)]
    valt = val.transpose(2, 3)
    for arg1 in range(0, vars):
        for arg2 in range(0, vars):
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
            i += 1
    return torch.cat(ret, dim=2)
   

def infer_single_step(ex_val : torch.Tensor, 
        body_predicates : torch.Tensor, variable_choices : torch.Tensor,
        rule_weights : torch.Tensor) -> torch.Tensor:
    logging.debug(f"{ex_val.shape=} {body_predicates.shape=} {variable_choices.shape=} {rule_weights.shape=}")
    ex_val = ex_val[:, body_predicates, variable_choices]
    #rule weighing
    rule_weights = rule_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #atoms
    rule_weights = rule_weights.unsqueeze(0) #worlds
    ex_val = ex_val#.unsqueeze(-5).unsqueeze(-5)
    logging.debug(f"{ex_val.shape=} {rule_weights.shape=}")
    ex_val = ex_val * rule_weights
    ex_val = ex_val.sum(dim = -4)
    ex_val = ex_val.where(ex_val.isnan().logical_not(), torch.zeros(size=(), device=ex_val.device)) #type: ignore
    #conjuction of body predictes
    ex_val = conjunction_dim(ex_val, -4)
    #existential quantification
    ex_val = disjunction_dim(ex_val, -1)
    #disjunction on clauses
    ex_val = disjunction_dim(ex_val, -3)
    logging.debug(f"returning {ex_val.shape=}")
    return ex_val

def infer_steps(steps : int, base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor, vars : int = 3) -> torch.Tensor:
    val = base_val
    vals : List[torch.Tensor] = []
    for i in range(0, steps):
        val2 = extend_val(val, vars)
        val2 = infer_single_step(ex_val = val2, body_predicates=rulebook.body_predicates, variable_choices=rulebook.variable_choices, \
            rule_weights = weights)
        assert val.shape == val2.shape, f"{i=} {val.shape=} {val2.shape=}"
        vals.append(val2.unsqueeze(0))
        val = disjunction2(val, val2)
    return disjunction_dim(torch.cat(vals), 0)
    return val
        
def loss(base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor,
        targets : torch.Tensor,
        target_values : torch.Tensor,
        steps : int = 2, vars : int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    val = infer_steps(steps, base_val, rulebook, weights, vars)
    preds = val[targets[:,0],targets[:,1],targets[:,2],targets[:,3]]
    #return (preds - target_values).square(), preds
    return (- ((preds + 1e-10).log() * target_values + (1-preds + 1e-10).log() * (1-target_values))), preds

def var_choices(n : int, vars : int = 3) -> List[int]:
    return [int(n) // vars, n % vars]

def rule_str(chosen_rules : List[int], clause : int, predicate : int, rulebook : Rulebook, pred_names : Dict[int,str]) -> str:
    ret = []
    for i, rule in enumerate(chosen_rules):
        vs = ','.join(map(lambda v: chr(ord('A')+v), var_choices(int(rulebook.variable_choices[predicate,clause,i,rule]))))
        ret.append(f'{pred_names[int(rulebook.body_predicates[predicate,clause,i,rule].item())]}({vs})')
    return ','.join(ret)

def print_program(rulebook : Rulebook, weights : torch.Tensor, pred_names : Dict[int,str], elements : int = 1):
    for pred, rules in enumerate(rulebook.body_predicates):
        if rules.numel() == 0:
            continue
        pred_name = pred_names[pred]
        if weights[pred].sum().item() == 0: continue
        wei = weights[pred].detach().cpu()
        for clause in range(0, 2):
            values, idxs = wei[clause].sort(-1,descending=True)
            ret = []
            for elem in range(0, elements):
                ret.append(rule_str([int(idxs[i][elem]) for i in range(2)], clause, pred, rulebook, pred_names) + '.')
            print(f"{pred_name.rjust(10, ' ')}(A,B) :- " + ' '.join(x.ljust(50, ' ') for x in ret))
