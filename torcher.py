from numpy import positive
import torch
from loader import Problem, World, rev_dict, TargetType
from typing import *
from dilp import Rulebook
import logging
import itertools

class TargetSet(NamedTuple):
    value : float
    idxs : torch.Tensor

    def to(self, device : torch.device) -> 'TargetSet':
        return TargetSet(
            value = self.value,
            idxs = self.idxs.to(device)
        )

    def __len__(self) -> int:
        return len(self.idxs)

class WorldsBatch(NamedTuple):
    base_val : torch.Tensor
    positive_targets : TargetSet
    negative_targets : TargetSet

    def targets(self, target_type : TargetType):
        if target_type == TargetType.POSITIVE:
            return self.positive_targets
        else:
            return self.negative_targets
    
    def to(self, device : torch.device) -> 'WorldsBatch':
        return WorldsBatch(
            base_val = self.base_val.to(device),
            positive_targets = self.positive_targets.to(device),
            negative_targets = self.negative_targets.to(device)
        )

T = TypeVar('T')

def chunks(n : int, seq : Sequence[T]) -> Iterable[Sequence[T]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def base_val(problem : Problem, worlds : Sequence[World]) -> torch.Tensor:
    atom_count = max(len(w.atoms) for w in worlds)
    ret = torch.zeros(size = [len(worlds), len(problem.predicate_name), atom_count, atom_count], dtype = torch.float)
    for i, world in enumerate(worlds):
        for fact in world.facts:
            ret[i][fact] = 1.0
    return ret

def targets_iter(worlds : Sequence[World], target_type : TargetType) -> Iterable[Sequence[int]]:
    for i, world in enumerate(worlds):
        targets = world.positive if target_type == TargetType.POSITIVE else world.negative
        for target in targets:
            yield [i] + list(target)

def targets(worlds : Sequence[World], target_type : TargetType) -> torch.Tensor:
    return torch.as_tensor(list(targets_iter(worlds, target_type)), dtype=torch.long)

def targets_batch(problem : Problem, worlds : Sequence[World], device : torch.device) -> WorldsBatch:
    positive_targets = targets(worlds, TargetType.POSITIVE)
    negative_targets = targets(worlds, TargetType.NEGATIVE)
    return WorldsBatch(
        base_val = base_val(problem, worlds).to(device),
        positive_targets = TargetSet(
            value = 1.0,
            idxs = positive_targets.to(device)
        ),
        negative_targets = TargetSet(
            value = 0.0,
            idxs = negative_targets.to(device)
        )
    )

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

def chv(n : int) -> str:
    return chr(ord('A')+n)

def rules(problem : Problem, 
            dev : torch.device,
            layers : Optional[List[int]] = None,
            unary : List[str] = [],
            recursion : bool = True, 
            invented_recursion : bool = True,
            full_rules : bool = False,
        ) -> Rulebook:

    if layers is None:
        layer_dict : Dict[int, int] = dict()
    else:
        layer_dict = dict(zip(problem.invented, sum(([i for _ in range(layer)] for i, layer in enumerate(layers)), start=[])))

    pred_dim = len(problem.predicate_name)

    unary_preds = set(problem.predicate_number[name] for name in unary)

    rev_pred = problem.predicate_name

    #ret = -torch.ones(size=(pred_dim, 2, 2, pred_dim * 3 * 3, 2), dtype=torch.long)

    ret = torch.zeros(size=(pred_dim, 2, 2, pred_dim * 3 * 3), dtype=torch.bool)

    for head in range(pred_dim):
        if head not in problem.bk:
            for clause in range(2):
                for body_position in range(2):
                    for i, (p, a, b) in enumerate(itertools.product(range(pred_dim),range(3),range(3))):
                        if not full_rules:
                            if p == head and a == 0 and b == 1: continue #self recursion
                            
                            if head in unary_preds and 1 in {a,b}: continue #using second arg of unary target
                            
                            if p in unary_preds and a != b: continue #calling unary with two different arguments
                            
                            #if any(head_pred in invented_preds and p in invented_preds and p < head_pred for p in {p1,p2}): continue

                            if not recursion and head == p: continue #recursion disabled

                            if not invented_recursion and head in problem.invented and p in {head, 0}: continue #recursion of inventeds disabled

                            #if head != 0 and p == 0: continue #THIS WAS USED IS SOME TESTS, MIGHT BE IMPORTANT?

                            if layers is not None and head in problem.invented and p != head and p in problem.invented and layer_dict[head]+1 != layer_dict[p]: continue

                            if layers is not None and head == 0 and p in problem.invented and layer_dict[p] != 0: continue #main pred only calls first layer

                        #ret[head,clause,body_position,i] = torch.as_tensor([p, a * 3 + b])
                        ret[head,clause,body_position,i] = True
                        logging.debug(f'rule {rev_pred[head]}(A,B) [{clause}] :- {"_, " if body_position == 1 else ""} {rev_pred[p]}({chv(a)}, {chv(b)})  {", _" if body_position == 0 else ""}')

    #cnt : int = int((ret >= 0).max(0)[0].max(0)[0].max(0)[0].max(1)[0].sum().item())
    #bp = ret[:,:,:,:cnt,0].to(dev)
    #vc = ret[:,:,:,:cnt,1].to(dev)

    #logging.info(f"{bp.shape=}")

    #if not full_rules:
    #    mask = (bp >= 0)
    #else:
    #    mask = torch.as_tensor([pred not in problem.bk for pred in range(pred_dim)], dtype=torch.bool, device=dev).unsqueeze(1).unsqueeze(1).unsqueeze(1)

    return Rulebook(
        mask = ret,
        full_rules = full_rules
    )