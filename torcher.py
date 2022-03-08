from numpy import positive
import torch
from loader import Problem, World, rev_dict
from typing import *
from dilp import Rulebook
import logging
import itertools

def base_val(problem : Problem, worlds : Sequence[World]) -> torch.Tensor:
    atom_count = max(len(w.atoms) for w in worlds)
    ret = torch.zeros(size = [len(worlds), len(problem.predicates), atom_count, atom_count], dtype = torch.float)
    for i, world in enumerate(worlds):
        for fact in world.facts:
            ret[i][fact] = 1.0
    return ret

def targets_iter(worlds : Sequence[World], positive : bool) -> Iterable[Sequence[int]]:
    for i, world in enumerate(worlds):
        targets = world.positive if positive else world.negative
        for target in targets:
            yield [i] + list(target)

def targets(worlds : Sequence[World], positive : bool) -> torch.Tensor:
    return torch.as_tensor(list(targets_iter(worlds, positive)), dtype=torch.long)

def targets_tuple(worlds : Sequence[World], device : torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    positive_targets = targets(worlds, positive = True)
    negative_targets = targets(worlds, positive = False)
    all_targets = torch.cat([positive_targets, negative_targets])
    target_values = torch.cat([
        torch.ones(len(positive_targets), device=device),
        torch.zeros(len(negative_targets), device=device)
    ])
    return all_targets, target_values

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

def rules(problem : Problem, 
            dev : torch.device,
            layers : Optional[List[int]] = None,
            unary : List[str] = [],
            recursion : bool = True, 
            invented_recursion : bool = True,
        ) -> Rulebook:

    if layers is None:
        layer_dict : Dict[int, int] = dict()
    else:
        layer_dict = dict(zip(problem.invented, sum(([i for _ in range(layer)] for i, layer in enumerate(layers)), start=[])))

    print(f"{len(problem.invented)=} {layer_dict=}")

    pred_dim = len(problem.predicates)

    unary_preds = set(problem.predicates[name] for name in unary)

    rev_pred = rev_dict(problem.predicates)

    ret = -torch.ones(size=(pred_dim, 2, 2, pred_dim * 3 * 3, 2), dtype=torch.long)

    for head in range(pred_dim):
        if head not in problem.bk:
            for clause in range(2):
                for body_position in range(2):
                    i = 0
                    for p in range(pred_dim):
                        for a, b in itertools.product(range(3),range(3)):
                            if p == head and a == 0 and b == 1: continue #self recursion
                            
                            if head in unary_preds and 1 in {a,b}: continue #using second arg of unary target
                            
                            if p in unary_preds and a != b: continue #calling unary with two different arguments
                            
                            #if any(head_pred in invented_preds and p in invented_preds and p < head_pred for p in {p1,p2}): continue

                            if not recursion and head == p: continue #recursion disabled

                            if not invented_recursion and head in problem.invented and p in {head, 0}: continue #recursion of inventeds disabled

                            #if head != 0 and p == 0: continue #THIS WAS USED IS SOME TESTS, MIGHT BE IMPORTANT?

                            if layers is not None and head in problem.invented and p != head and p in problem.invented and layer_dict[head]+1 != layer_dict[p]: continue

                            if layers is not None and head == 0 and p in problem.invented and layer_dict[p] != 0: continue #main pred only calls first layer

                            ret[head,clause,body_position,i] = torch.as_tensor([p, a * 3 + b])
                            i += 1
                            logging.debug(f'rule {rev_pred[head]} [{clause}] :- {"_, " if body_position == 1 else ""} {rev_pred[p]}({a}, {b})  {", _" if body_position == 0 else ""}')

    cnt : int = int((ret >= 0).max(0)[0].max(0)[0].max(0)[0].max(1)[0].sum().item())
    bp = ret[:,:,:,:cnt,0].to(dev)
    vc = ret[:,:,:,:cnt,1].to(dev)

    logging.info(f"{bp.shape=} {cnt=}")

    return Rulebook(
        body_predicates = bp,
        variable_choices = vc,
        mask = (bp >= 0)
    )