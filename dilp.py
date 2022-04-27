from sys import exec_prefix
import torch
import logging
import math
import loader
from typing import *

from zmq import device
import weird

class Rulebook(NamedTuple):
    mask : torch.Tensor #boolean, true if rule is used

    def to(self, device : torch.device, non_blocking : bool = True):
        return Rulebook(
            mask=self.mask.to(device, non_blocking=non_blocking)
        )

def disjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a + b - a * b
def disjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    if a.shape[dim] == 2:
        disjunction2_prod(*a.split(1, dim=dim))
    return 1 - ((1 - a).prod(dim))
def conjunction2_prod(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return a * b
def conjunction_dim_prod(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return a.prod(dim=dim)

def disjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.max(a, b)
    return a.where(a>b, b)
def disjunction_dim_max(a : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.max(a, dim=dim)[0]
def conjunction2_max(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    #return torch.min(a, b)
    return a.where(a<b, b)
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
        rule_weights : torch.Tensor) -> torch.Tensor:
    logging.debug(f"{ex_val.shape=} {rule_weights.shape=}")

    shape = list(ex_val.shape)
    shape[1] *= shape[2]
    del shape[2]
    ex_val = ex_val.reshape(shape)
    ex_val = ex_val.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    #rule weighing
    rule_weights = rule_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #atoms
    rule_weights = rule_weights.unsqueeze(0) #worlds
    ex_val = ex_val#.unsqueeze(-5).unsqueeze(-5)
    logging.debug(f"{ex_val.shape=} {rule_weights.shape=}")
    ex_val = ex_val * rule_weights
     
    ex_val = ex_val.sum(dim = -4)
    #ex_val = ex_val.where(ex_val.isnan().logical_not(), torch.zeros(size=(), device=ex_val.device)) #type: ignore
    
    control_valve = torch.max(ex_val.detach()-1, torch.zeros(size=(), device=ex_val.device))
    ex_val = ex_val - control_valve
    #conjuction of body predictes
    ex_val = conjunction_dim(ex_val, -4)
    #existential quantification
    ex_val = disjunction_dim(ex_val, -1)
    #disjunction on clauses
    ex_val = disjunction_dim(ex_val, -3)
    logging.debug(f"returning {ex_val.shape=}")
   
    return ex_val

def infer_steps_on_devs(steps : int, base_val : torch.Tensor,
        return_dev : torch.device, devices : Sequence[torch.device],
        rule_weights : torch.Tensor,
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
                rule_weights = rule_weights_[i]))
        val = disjunction2(val, torch.cat([t.to(return_dev, non_blocking=True) for t in rets], dim=1))
    
    return val



def infer_steps(steps : int, base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor, vars : int = 3) -> torch.Tensor:
    val = base_val
    #vals : List[torch.Tensor] = []
    for i in range(0, steps):
        val2 = extend_val(val, vars)
        val2 = infer_single_step(ex_val = val2, \
            rule_weights = weights)
        assert val.shape == val2.shape, f"{i=} {val.shape=} {val2.shape=}"
        #vals.append(val2.unsqueeze(0))
        val = disjunction2(val, val2)
    #return disjunction_dim(torch.cat(vals), 0)
    return val

def infer(base_val : torch.Tensor,
            rulebook : Rulebook,
            weights : torch.Tensor,
            steps : int,
            devices : Optional[Sequence[torch.device]] = None,
            ) -> torch.Tensor:
    if devices is None:
        return infer_steps(steps, base_val, rulebook, weights, 3)
    else:
        return infer_steps_on_devs(steps, base_val, weights.device, devices,
            weights)

def loss(values : torch.Tensor, target_type : loader.TargetType, reduce : bool = True) -> torch.Tensor:
    if target_type == loader.TargetType.POSITIVE:
        ret = -(values + 1e-10).log()
        if reduce:
            ret = ret.mean()
        return ret
    else:
        return loss(1-values, target_type=loader.TargetType.POSITIVE, reduce=reduce)

def extract_targets(vals : torch.Tensor, targets : torch.Tensor) -> torch.Tensor:
    return vals[targets[:,0],targets[:,1],targets[:,2],targets[:,3]]

def legacy_loss(base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor,
        targets : torch.Tensor,
        target_values : torch.Tensor, steps : int, 
        devices : Optional[Sequence[torch.device]] = None,
        vars : int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    if devices is None:
        val = infer_steps(steps, base_val, rulebook, weights, vars)
    else:
        val = infer_steps_on_devs(steps, base_val, devices[-1], devices, weights)
    preds = val[targets[:,0],targets[:,1],targets[:,2],targets[:,3]]
    #return (preds - target_values).square(), preds
    return (- ((preds + 1e-10).log() * target_values + (1-preds + 1e-10).log() * (1-target_values))), preds

def var_choices(n : int, vars : int = 3) -> List[int]:
    return [int(n) // vars, n % vars]

def body_pred_str(pred_name : str, variable_choice : int) -> str:
    vs = ','.join(map(lambda v: chr(ord('A')+v), var_choices(variable_choice)))
    return f'{pred_name}({vs})'

def print_program(problem : loader.Problem, weights : torch.Tensor):
    for pred in list(problem.targets) + list(problem.invented):
        pred_name = problem.predicate_name[pred]
        for clause in range(2):
            body_preds = []
            for body_pred in range(2):
                rule_no = int(weights[pred,clause,body_pred].max(0)[1].item())
                rule_pred = rule_no // 9
                rule_vc = rule_no % 9
                body_preds.append(body_pred_str(problem.predicate_name[rule_pred], rule_vc))
            print(f"{pred_name.rjust(10, ' ')}(A,B) :- " + ','.join(body_preds))