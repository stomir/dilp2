from numpy import positive
import torch
from loader import Problem, World, rev_dict, TargetType
from typing import *
import logging
import itertools
import random
import numpy

base_filter = filter

class TargetSet(NamedTuple):
    value : float
    idxs : torch.Tensor # world, predicate, x, y

    def to(self, device : torch.device) -> 'TargetSet':
        return TargetSet(
            value = self.value,
            idxs = self.idxs.to(device)
        )

    def __len__(self) -> int:
        return len(self.idxs)
    
    def filter(self, f : Callable[[Sequence[int]], bool]) -> 'TargetSet':
        if len(self.idxs) == 0:
            return self
        idxs = self.idxs.cpu().numpy()
        return TargetSet(
            value = self.value,
            idxs = torch.from_numpy(idxs[numpy.array([f(t) for t in idxs])]).to(self.idxs.device)
        )

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
        
    def filter(self, f : Callable[[Sequence[int]], bool]) -> 'WorldsBatch':
        return WorldsBatch(
            base_val=self.base_val,
            positive_targets=self.positive_targets.filter(f),
            negative_targets=self.negative_targets.filter(f)
        )
        
class Rulebook(NamedTuple):
    mask : torch.Tensor #boolean, true if rule is used

    def to(self, device : torch.device, non_blocking : bool = True):
        return Rulebook(
            mask=self.mask.to(device, non_blocking=non_blocking)
        )

T = TypeVar('T')
def chunks(n : int, seq : Sequence[T]) -> Iterable[Sequence[T]]:
    """
    splits the sequence into chunks of length `n`

    Args:
        n (int): length of chunks
        seq (Sequence[T]): input sequence

    Returns:
        Iterable[Sequence[T]]: generator of sequences of length `n`

    """
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def base_val(problem : Problem, worlds : Sequence[World], dtype) -> torch.Tensor:
    """
    creates truth tensor with background knowledge of a problem

    Args:
        problem (Problem): problem description
        worlds (Sequence[World]): worlds to be included
        dtype (_type_): type to returned tensor

    Returns:
        torch.Tensor: truth tensor with background knowledge of a problem
    """
    atom_count = max(len(w.atoms) for w in worlds)
    ret = torch.zeros(size = [len(worlds), len(problem.predicate_name), atom_count, atom_count], dtype = dtype)
    for i, world in enumerate(worlds):
        for fact in world.facts:
            ret[i][fact] = 1.0
        if '$true' in problem.predicate_number:
            ret[i][problem.predicate_number['$true']] = 1.0
    return ret

def targets_iter(worlds : Sequence[World], target_type : TargetType) -> Iterable[Sequence[int]]:
    """
    an iterable of lists of coordinates of training examples

    Args:
        worlds (Sequence[World]): worlds to get targets for
        target_type (TargetType): whether to get positive of negative examples

    Returns:
        Iterable[Sequence[int]]: coordinates of training examples
    """
    for i, world in enumerate(worlds):
        targets = world.positive if target_type == TargetType.POSITIVE else world.negative
        for target in targets:
            yield [i] + list(target)

def targets(worlds : Sequence[World], target_type : TargetType) -> torch.Tensor:
    """
    a tensor containing coordinates of training examples

    Args:
        worlds (Sequence[World]): worlds to get targets for
        target_type (TargetType): whether to get positive of negative examples

    Returns:
        torch.Tensor: coordinates of training examples
    """
    return torch.as_tensor(list(targets_iter(worlds, target_type)), dtype=torch.long)

def targets_batch(problem : Problem, worlds : Sequence[World], device : torch.device, dtype) -> WorldsBatch:
    """
    prepare a batch out of a sequence of worlds

    Args:
        problem (Problem): problem description
        worlds (Sequence[World]): sequence of worlds
        device (torch.device): what device to put output tensors on
        dtype (_type_): what type should tensors be

    Returns:
        WorldsBatch: prepared batch of worlds
    """
    positive_targets = targets(worlds, TargetType.POSITIVE)
    negative_targets = targets(worlds, TargetType.NEGATIVE)
    return WorldsBatch(
        base_val = base_val(problem, worlds, dtype = dtype).to(device),
        positive_targets = TargetSet(
            value = 1.0,
            idxs = positive_targets.to(device)
        ),
        negative_targets = TargetSet(
            value = 0.0,
            idxs = negative_targets.to(device)
        )
    )

def rules(problem : Problem, 
            dev : torch.device,
            split : int = 2,
            unary : List[str] = [],
            recursion : bool = True, 
            invented_recursion : bool = True,
            allow_cross_targets : bool = True,
            full_rules : bool = False,
        ) -> Rulebook:
    """
    prepares rule weights, optionally applying a language bias mask
    
    note: language bias arguments only work for split 2

    Args:
        problem (Problem): problem description
        dev (torch.device): device to put tensors on
        split (int, optional): how is the program split. Defaults to 2.
        unary (List[str], optional): list of predicates to be treated as unary. Defaults to [].
        recursion (bool, optional): should recusion be allowed. Defaults to True.
        full_rules (bool, optional): overrides and removed all language bias. Defaults to False.
        allow_cross_targets (bool, optional): whether to allow target copies to call each other. Defaults to True.

    Raises:
        NotImplementedError: if given split other than {0,1,2}

    Returns:
        Rulebook: prepared rulebook
    """
        
    pred_dim = len(problem.predicate_name)

    unary_preds = set(problem.predicate_number[name] for name in unary)

    rev_pred = problem.predicate_name
                
    parent_target : Dict[int, int] = dict()
    for original, copies in problem.target_copies.items():
        parent_target[original] = original
        for copy in copies:
            parent_target[copy] = original

    if split == 2:

        ret = torch.ones(size=(pred_dim, 2, 2, pred_dim * 3 * 3), dtype=torch.bool)

        if not full_rules:
            for head in range(pred_dim):
                if head not in problem.bk:
                    for clause in range(2):
                        for body_position in range(2):
                            for i, (p, a, b) in enumerate(itertools.product(range(pred_dim),range(3),range(3))):
                                    if ((not p == head and a == 0 and b == 1) #self recursion
                                        or (head in unary_preds and 1 in {a,b}) #using second arg of unary target
                                        or (p in unary_preds and a != b) #calling unary with two different arguments
                                        or (not recursion and head == p) #recursion disabled
                                        or (not invented_recursion and head in problem.invented and p in {head, 0}) #recursion of inventeds disabled
                                        or (not allow_cross_targets and head != p and head in parent_target and p in parent_target and parent_target[head] == parent_target[p])
                                        or (not allow_cross_targets and head in problem.invented and p in parent_target)):

                                        ret[head,clause,body_position,i] = False
    elif split == 1:
        ret = torch.ones(size=(pred_dim, 2, (pred_dim * 3 * 3) ** 2), dtype=torch.bool)

        if not full_rules:
            for head in range(pred_dim):
                if head not in problem.bk:
                    for clause in range(2):
                        for i, (p1, a1, b1, p2, a2, b2) in enumerate(itertools.product(range(pred_dim),range(3),range(3),range(pred_dim),range(3),range(3))):
                            if (p1, a1, b1) == (head, 0, 1) \
                                or (p2, a2, b2) == (head, 0, 1): 
                                    ret[head,clause,i] = False #self recursion

    elif split == 0:
        ret = torch.ones(size=(pred_dim, (pred_dim * 3 * 3) ** 4), dtype=torch.bool)

        if not full_rules:
            for head in range(pred_dim):
                if head not in problem.bk:
                    for clause in range(2):
                        for i, (c1p1, c1a1, c1b1, c1p2, c1a2, c1b2, c2p1, c2a1, c2b1, c2p2, c2a2, c2b2) \
                                in enumerate(itertools.product(range(pred_dim),range(3),range(3),range(pred_dim),range(3),range(3),
                                                                                range(pred_dim),range(3),range(3),range(pred_dim),range(3),range(3))):
                            if (c1p1, c1a1, c1b1) == (head, 0, 1) \
                                or (c2p1, c2a1, c2b1) == (head, 0, 1) \
                                or (c2p2, c2a2, c2b2) == (head, 0, 1) \
                                or (c1p2, c1a2, c1b2) == (head, 0, 1): 
                                    ret[head,i] = False #self recursion

    else:
        raise NotImplementedError(f'wrong {split=}')
                        

    return Rulebook(
        mask = ret.to(dev)
    )