import torch
import logging
from typing import *

class Rulebook(NamedTuple):
    body_predicates : torch.Tensor
    variable_choices : torch.Tensor

    def to(self, device : torch.device) -> 'Rulebook':
        return Rulebook(
            body_predicates = self.body_predicates.to(device),
            variable_choices = self.variable_choices.to(device))
#'''
def disjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return 1 - (1 - a) * (1 - b)

def disjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return 1 - ((1 - a).prod(dim = dim))

def conjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a * b
def conjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.prod(dim=dim)
#'''

#'''
def disjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.max(a, b)

def disjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.max(a, dim=dim)[0]

def conjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.min(a, b)

def conjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.min(dim=dim)[0]
#'''

disjunction2 = disjunction2_max
disjunction_dim = disjunction_dim_max
conjunction2 = conjunction2_max
conjunction_dim = conjunction_dim_max

def set_norm(norm_name : str):
    global disjunction2, disjunction_dim, conjunction2, conjunction_dim
    if norm_name == 'max':
        disjunction2 = disjunction2_max
        disjunction_dim = disjunction_dim_max
        conjunction2 = conjunction2_max
        conjunction_dim = conjunction_dim_max
    elif norm_name == 'prod':
        disjunction2 = disjunction2_max
        disjunction_dim = disjunction_dim_max
        conjunction2 = conjunction2_max
        conjunction_dim = conjunction_dim_max

def var_choices(n : int, vars : int = 3) -> List[int]:
    return [int(n) // vars, n % vars]

def rule_str(rs : Sequence[int], predicate : int, rulebook : Rulebook, pred_names : Dict[int,str]) -> str:
    lines = []
    for clause in range(0, rulebook.body_predicates.shape[1]):
        ret = []
        for i in range(0, rulebook.body_predicates.shape[3]):
            vs = ','.join(map(lambda v: chr(ord('A')+v),  var_choices(int(rulebook.variable_choices[predicate,clause,rs[clause],i]))))
            ret.append(f'{pred_names[rulebook.body_predicates[predicate,clause,rs[clause],i].item()]}({vs})')
        lines.append(f"{pred_names[predicate]}(A,B) :- {','.join(ret)}")
    return '\n'.join(lines)

def extend_val(val : torch.Tensor, vars : int = 3) -> torch.Tensor:
    i = 0
    ret = []
    shape = list(val.shape) + [val.shape[-1] for _ in range(0, vars - 2)]
    valt = val.transpose(1, 2)
    for arg1 in range(0, vars):
        for arg2 in range(0, vars):
            v = val
            if arg1 == arg2:
                v = v.diagonal(dim1=1,dim2=2)
                for _ in range(0, arg1):
                    v = v.unsqueeze(1)
                while len(v.shape) < vars+1:
                    v = v.unsqueeze(-1)
            else:
                if arg2 < arg1:
                    v = valt
                unused = (x for x in range(0, vars) if x not in {arg1, arg2})
                for u in unused:
                    v = v.unsqueeze(u + 1)
            logging.debug(f"{i=} {arg1=} {arg2=} {v.shape=}")
            v = torch.broadcast_to(v, shape) #type: ignore
            v = v.unsqueeze(1)
            ret.append(v)
            i += 1
    return torch.cat(ret, dim=1)

def infer_single_step(ex_val : torch.Tensor, rules : Rulebook, rule_weights : torch.Tensor) -> torch.Tensor:
    logging.debug(f"{ex_val.shape=} {rules.body_predicates.shape=} {rules.variable_choices.shape=}")
    ex_val = ex_val[rules.body_predicates, rules.variable_choices]
    logging.debug(f"{ex_val.shape=}")
    #conjuction of body predictes
    ex_val = conjunction_dim(ex_val, dim = 3)
    #existential quantification
    ex_val = disjunction_dim(ex_val, dim = -1)
    #rule weighing
    rule_weights = rule_weights.softmax(-1).unsqueeze(-1).unsqueeze(-1)
    ex_val = ex_val * rule_weights
    ex_val = ex_val.sum(dim = 2)
    #disjunction on clauses
    ex_val = disjunction_dim(ex_val, dim = 1)
    return ex_val
        
def loss(base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor,
        targets : torch.Tensor,
        target_values : torch.Tensor,
        steps : int = 2, vars : int = 3) -> torch.Tensor:
    val = base_val
    vals = []
    for i in range(0, steps):
        val2 = extend_val(val, vars)
        val2 = infer_single_step(val2, rulebook, weights)
        vals.append(val2.unsqueeze(0))
        val = disjunction2(val, val2)
    val = disjunction_dim(torch.cat(vals, dim=0), dim=0)
    preds = val[targets[:,0],targets[:,1],targets[:,2]]
    return (preds - target_values).square().mean()
    
def print_program(rulebook : Rulebook, weights : torch.Tensor, pred_names : List[str]):
    weights = weights.detach().cpu()
    for pred in range(0, rulebook.body_predicates.shape[0]):
        print(rule_str(rs = weights[pred].max(dim = -1)[1].numpy(), predicate=pred, rulebook=rulebook, pred_names=pred_names))
