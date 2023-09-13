import torch
import logging
import math
import loader
import itertools
from typing import *
from loader import Problem
from torcher import WorldsBatch, TargetSet, TargetType, Rulebook
import torch.nn.functional as F
import weird

# different norms
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

# this isn't softmax, it's one of experiemntal norms
def soft_max(t : torch.Tensor, d : int, p : float) -> torch.Tensor:
    return (t * (t * p).softmax(dim=d)).sum(dim=d)

# helper function for changing norm functions to other formats
def dim2two(f : Callable[[torch.Tensor, int], torch.Tensor]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return lambda a,b: f(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), 0)

def two2dim(f : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def do(t : torch.Tensor, dim : int):
        l = [u.squeeze(dim) for u in t.split(split_size=1, dim=dim)]
        while len(l) > 1:
            ret = []
            while len(l) > 1:
                a = l.pop()
                b = l.pop()
                ret.append(f(a,b))
            if len(l) > 0:
                ret.append(l.pop())
            l = ret
        return l[0]
    return do
        
class Norms(NamedTuple):
    """
    a tuple containing norms to be using in differentiable inference
    """
    conjunction_body_pred : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = conjunction2_max
    disjunction_quantifier : Callable[[torch.Tensor, int], torch.Tensor] = disjunction_dim_max
    disjunction_steps : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  = disjunction2_max
    disjunction_clauses : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = disjunction2_max

    @staticmethod
    def from_name(norm_name : str = 'max', p : float = 1.0) -> 'Norms':
        """
        A function for getting a norms tuple from a name

        Args:
            norm_name (str, optional): name of a norm. Defaults to 'max'.
            p (float, optional): some norms use a parameter in their definition. Defaults to 1.0.

        Raises:
            NotImplementedError: when given an unknown name

        Returns:
            Norms: a tuple containing norms to be using in differentiable inference
        """
        if norm_name == 'max':
            return Norms(conjunction_body_pred = conjunction2_max,
                disjunction_quantifier = disjunction_dim_max,
                disjunction_steps = disjunction2_max,
                disjunction_clauses = disjunction2_max)
        elif norm_name == 'prod':
            return Norms(conjunction_body_pred = conjunction2_prod,
                disjunction_quantifier = disjunction_dim_prod,
                disjunction_steps = disjunction2_prod,
                disjunction_clauses = disjunction2_prod)
        elif norm_name == 'mixed':
            return Norms(conjunction_body_pred = conjunction2_prod,
                disjunction_quantifier = disjunction_dim_max,
                disjunction_steps = disjunction2_max,
                disjunction_clauses = disjunction2_max)
        elif norm_name == 'dilp1':
            return Norms(conjunction_body_pred = conjunction2_prod,
                disjunction_quantifier = disjunction_dim_max,
                disjunction_steps = disjunction2_prod,
                disjunction_clauses = disjunction2_max)
        elif norm_name == 'soft_max_ex':
            return Norms(conjunction_body_pred = conjunction2_prod,
                disjunction_quantifier = lambda t,d: soft_max(t,d,p),
                disjunction_steps = disjunction2_max,
                disjunction_clauses = disjunction2_max)
        elif norm_name == 'soft_max':
            return Norms(conjunction_body_pred = dim2two(lambda t,d: soft_max(t,d,-p)),
                disjunction_quantifier = lambda t,d: soft_max(t,d,p),
                disjunction_steps = dim2two(lambda t,d: soft_max(t,d,p)),
                disjunction_clauses = dim2two(lambda t,d: soft_max(t,d,p)))
        else:
            raise NotImplementedError(f"wrong norm name {norm_name=}")

#different ways of initializing weights
def random_init(init : str, device : torch.device, shape : List[int], init_size : float, dtype) -> torch.Tensor:
    """
    function for initializing weights

    Args:
        init (str): one of "normal", "uniform" and "discrete"
        device (torch.device): a device on which the weights should be stored
        shape (List[int]): shape
        init_size (float): a parameter scaling the weights
        dtype (_type_): type for the weights

    Raises:
        RuntimeError: when given a wrong init string

    Returns:
        torch.Tensor: random weights
    """
    if init == 'normal':
        return torch.normal(mean=torch.zeros(size=shape, device=device, dtype=dtype), std=init_size)
    elif init == 'uniform':
        return torch.rand(size=shape, device=device, dtype=dtype) * init_size
    elif init == 'discrete':
        return F.one_hot(torch.randint(low=0, high=shape[-1], size=shape[:-1], device=device), num_classes = shape[3]).float()
    else:
        raise RuntimeError(f'unknown init: {init}')

def masked_softmax(t : torch.Tensor, mask : torch.Tensor, temp : Optional[float]) -> torch.Tensor:
    """
    computes a masked softmax (can handle entire rows being masked - then output is 0s)

    Args:
        t (torch.Tensor): input
        mask (torch.Tensor): mask
        temp (Optional[float]): temperature

    Returns:
        torch.Tensor: masked softmax
    """
    if temp is None or temp == 0.:
        t = t.where(mask, torch.as_tensor(0., device=t.device))
        if temp is not None:
            print(f'1 {t=}')
            t = t / torch.max(t.sum(dim = -1, keepdim=True), torch.as_tensor(1e-3, device=t.device))
            print(f'2 {t=}')
    else:
        t = t.where(mask, torch.as_tensor(-float('inf'), device=t.device)).softmax(-1)
        t = t.where(t.isnan().logical_not(), torch.as_tensor(0.0, device=t.device)) #type: ignore
    return t

def mask(tensor : torch.Tensor, rulebook : Rulebook) -> torch.Tensor:
    """
    applies a mask from a rulebook on given weights tensor

    Args:
        tensor (torch.Tensor): weights
        rulebook (Rulebook): rulebook

    Returns:
        torch.Tensor: masked weights
    """
    return tensor.where(rulebook.mask, torch.zeros(size=(),device=tensor.device))

class DILP(torch.nn.Module):
    """
    a module performing the inference

    Exists mainly to allow using `torch.compile`

    """
    def __init__(self, norms : Norms, 
                device : torch.device,
                devices : Optional[List[torch.device]],
                rulebook : Rulebook,
                init_type : str,
                init_size : float,
                steps : int,
                split : int,
                softmax_temp : Optional[float],
                problem : Problem
                ):
        super().__init__()
        shape = rulebook.mask.shape
        self.weights = torch.nn.Parameter(random_init(init_type, device = device, shape = list(shape), init_size = init_size, dtype = torch.float32))
        self.rulebook = rulebook
        self.problem = problem
        self.norms = norms
        self.steps = steps
        self.split = split
        self.devices = devices
        self.softmax_temp = softmax_temp

    def forward(self, base_val : torch.Tensor, crisp : bool = False) -> torch.Tensor:
        if crisp:
            w = mask(torch.nn.functional.one_hot(self.weights.max(-1)[1], self.weights.shape[-1]).to(base_val.device).float(), self.rulebook)
        else:
            w = masked_softmax(self.weights, self.rulebook.mask, temp = self.softmax_temp)
        return infer(base_val,
            rulebook=self.rulebook,
            weights=w,
            problem=self.problem,
            steps=self.steps,
            split=self.split,
            norms=self.norms,
            devices=self.devices)


def squeeze_into(v : torch.Tensor, dim : int, dim2 : int) -> torch.Tensor:
    """
    a function for mergin two dimensions of a tensor into one

    Args:
        v (torch.Tensor): tensor
        dim (int): first dimension
        dim2 (int): second dimension (will be removed, must be next to `dim`)

    Returns:
        torch.Tensor: _description_
    """
    assert abs(dim - dim2) == 1
    shape = list(v.shape)
    shape[dim] *= shape[dim2]
    del shape[dim2]
    return v.reshape(shape)

def extend_val(val : torch.Tensor) -> torch.Tensor:
    """ 
    extends a truth tensor of binary predicates into 9x more tertiary onces (one for each variable choice)

    Args:
        val (torch.Tensor): truth tensor

    Returns:
        torch.Tensor: extended truth tensor
    """
    ret = []
    shape = list(val.shape) + [val.shape[-1]]
    valt = val.transpose(2, 3)
    for i, (arg1, arg2) in enumerate(itertools.product(range(3), range(3))):
        v = val
        if arg1 == arg2:
            v = v.diagonal(dim1=2,dim2=3)
            for _ in range(0, arg1):
                v = v.unsqueeze(2)
            while len(v.shape) < 5:
                v = v.unsqueeze(-1)
        else:
            if arg2 < arg1:
                v = valt
            unused = (x for x in range(0, 3) if x not in {arg1, arg2})
            for u in unused:
                v = v.unsqueeze(u + 2)
        v = torch.broadcast_to(v, shape) 
        v = v.unsqueeze(2)
        ret.append(v)
    return torch.cat(ret, dim=2)

def infer_single_step(ex_val : torch.Tensor, 
        rule_weights : torch.Tensor,
        split : int,
        norms : Norms,
        ) -> torch.Tensor:
    """
    performs a single step of differetiable inference

    Args:
        ex_val (torch.Tensor): extend truth tensor
        rule_weights (torch.Tensor): weights
        split (int): how is the program split (one of 0,1,2)
        norms (Norms): tuple with norm functions

    Raises:
        NotImplementedError: when given wrong `split`

    Returns:
        torch.Tensor: truth tensor (no longer extended)
    """

    #add dimensions for atoms
    rule_weights = rule_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #add dimensions for worlds
    rule_weights = rule_weights.unsqueeze(0)

    #merge dimensions for predicates and variable choices
    ex_val = squeeze_into(ex_val, 1, 2)

    if split == 2:
        logging.debug(f"{ex_val.shape=} {rule_weights.shape=}")

        #add dimensions for rule weights (predicate, clause, body pred.)
        ex_val = ex_val.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        #weighted average by weights
        ex_val = ex_val * rule_weights 
        ex_val = ex_val.sum(dim = -4)

        #cut values above 1. without changing gradients
        ex_val = weird.WeirdMin1.apply(ex_val, torch.as_tensor(1.0, device=ex_val.device))

        # conjuction of body predictes
        ex_val = norms.conjunction_body_pred(ex_val[:,:,:,0,:,:,:], ex_val[:,:,:,1,:,:,:])
        # existential quantification
        ex_val = norms.disjunction_quantifier(ex_val, -1)
        # disjunction on clauses
        ex_val = norms.disjunction_clauses(ex_val[:,:,0,:,:], ex_val[:,:,1,:,:])
        logging.debug(f"returning {ex_val.shape=}")
        return ex_val
    elif split == 1:
        #conjuction of body predictes
        ex_val = norms.conjunction_body_pred(ex_val.unsqueeze(1), ex_val.unsqueeze(2))
        ex_val = squeeze_into(ex_val, 1, 2)
        
        #add dimensions for rule weights (predicate, clause)
        ex_val = ex_val.unsqueeze(0).unsqueeze(0)
        
        #weighted average by weights
        ex_val = ex_val * rule_weights 
        ex_val = ex_val.sum(dim = -4)

        #existential quantification
        ex_val = norms.disjunction_quantifier(ex_val, -1)
        #disjunction on clauses
        ex_val = norms.disjunction_clauses(ex_val[:,:,0,:,:], ex_val[:,:,1,:,:])
        logging.debug(f"returning {ex_val.shape=}")
        return ex_val
    elif split == 0:
        #conjuction of body predictes
        ex_val = norms.conjunction_body_pred(ex_val.unsqueeze(1), ex_val.unsqueeze(2))
        ex_val = squeeze_into(ex_val, 1, 2)

        #existential quantification
        ex_val = norms.disjunction_quantifier(ex_val, -1)
        rule_weights = rule_weights.squeeze(-1)

        #disjunction on clauses
        ex_val = norms.disjunction_clauses(ex_val.unsqueeze(1), ex_val.unsqueeze(2))
        ex_val = squeeze_into(ex_val, 1, 2)

        #add dimensions for rule weights (predicate,)
        ex_val = ex_val.unsqueeze(0)
        
        #weighted average by weights
        ex_val = ex_val * rule_weights
        ex_val = ex_val.sum(dim = -3)

        logging.debug(f"returning {ex_val.shape=}")
        return ex_val
    else:
        raise NotImplementedError(f"{split=}")


def infer_steps_on_devs(steps : int, base_val : torch.Tensor,
        problem : Problem,
        return_dev : torch.device, devices : Sequence[torch.device],
        rule_weights : torch.Tensor,
        split : int,
        norms : Norms,
        ) -> torch.Tensor:
    """
    performs multiple steps of inference using multiple GPUs in parallel

    Args:
        steps (int): number of steps
        base_val (torch.Tensor): starting truth tensor (background knowledge)
        problem (Problem): problem description
        return_dev (torch.device): device on which returns value should be
        devices (Sequence[torch.device]): GPUs to use
        rule_weights (torch.Tensor): weights
        split (int): how to split the program
        norms (Norms): norms to use

    Returns:
        torch.Tensor: truth tensor after `steps`
    """
    
    rule_weights = rule_weights[len(problem.bk):]
    bk_zeros = torch.zeros([base_val.shape[0], len(problem.bk), base_val.shape[2], base_val.shape[3]], device = return_dev)
    
    pred_count : int = rule_weights.shape[0]
    per_dev = math.ceil(pred_count / len(devices))
    print(f'{pred_count=} {per_dev=}')
    
    rule_weights_ : List[torch.Tensor] = []
    for i, dev in enumerate(devices):
        rule_weights_.append(rule_weights[i*per_dev:(i+1)*per_dev].to(dev, non_blocking=True))

    val = base_val
    for step in range(steps):
        rets = [bk_zeros]
        for i, dev in enumerate(devices):
            rets.append(infer_single_step(
                ex_val = extend_val(val.to(dev, non_blocking=True)),
                split=split,
                norms=norms,
                rule_weights = rule_weights_[i]))
        val = norms.disjunction_steps(val, torch.cat([t.to(return_dev, non_blocking=True) for t in rets], dim=1))
    
    return val

def infer_steps(steps : int, base_val : torch.Tensor, rulebook : Rulebook, weights : torch.Tensor, 
            problem : Problem,
            split : int,
            norms : Norms,
            ) -> torch.Tensor:
    """
    performs multiple steps of inference on a single device

    Args:
        steps (int): number of steps
        base_val (torch.Tensor): starting truth tensor (background knowledge)
        problem (Problem): problem description
        weights (torch.Tensor): rule weights
        problem (Problem): problem description
        split (int): how to split the program
        norms (Norms): norms to use

    Returns:
        torch.Tensor: _description_
    """
    val = base_val #non-nk weights

    weights = weights[len(problem.bk):]
    bk_zeros = torch.zeros([val.shape[0], len(problem.bk), val.shape[2], val.shape[3]], device = val.device)

    for i in range(0, steps):
        val2 = extend_val(val)
        val2 = infer_single_step(ex_val = val2, \
            split=split,
            norms=norms,
            rule_weights = weights)
        val2 = torch.cat([bk_zeros, val2], dim=1)
        
        assert val.shape == val2.shape, f"{i=} {val.shape=} {val2.shape=}"
        val = norms.disjunction_steps(val, val2)
    return val

def infer(base_val : torch.Tensor,
            rulebook : Rulebook,
            weights : torch.Tensor,
            problem : Problem,
            steps : int,
            split : int,
            norms : Norms,
            devices : Optional[Sequence[torch.device]] = None,
            ) -> torch.Tensor:
    """
    runs one of `infer_steps` or `infer_steps_on_devs`
    """
    if devices is None:
        return infer_steps(steps, base_val, rulebook, weights, problem, split=split, norms=norms)
    else:
        return infer_steps_on_devs(steps, base_val, problem, weights.device, devices, split = split,
            rule_weights=weights, norms=norms)

def loss_value(values : torch.Tensor, target_type : loader.TargetType, reduce : bool = True) -> torch.Tensor:
    """
    computes loss value given valuations

    Args:
        values (torch.Tensor): valuations
        target_type (loader.TargetType): are values supposed to be true or false
        reduce (bool, optional): whether the output should be reduce to a single number. Defaults to True.

    Returns:
        torch.Tensor: loss value
    """
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
    """
    extracts values from coordinates given in `targets` tensor, optionally filtering them

    Args:
        vals (torch.Tensor): values
        targets (torch.Tensor): coordinates
        filter (Optional[Callable[[Tuple[int,int,int]], bool]], optional): filter function. Defaults to None.

    Returns:
        torch.Tensor: extracted values
    """
    if filter is not None:
        targets = list(base_filter(filter, targets)) #type: ignore
    if len(targets) == 0:
        return torch.as_tensor([], device=targets.device)
    return vals[targets[:,0],targets[:,1],targets[:,2],targets[:,3]]

def loss(vals : torch.Tensor, batch : WorldsBatch, filter : Optional[Callable[[Tuple[int,int,int]], bool]] = None) -> torch.Tensor:
    """
    computes loss for a given batch, optionally using a filter

    Args:
        vals (torch.Tensor): values
        batch (WorldsBatch): batch
        filter (Optional[Callable[[Tuple[int,int,int]], bool]], optional): filter. Defaults to None.

    Returns:
        torch.Tensor: loss value
    """
    pos = extract_targets(vals, batch.positive_targets.idxs, filter = filter)
    neg = extract_targets(vals, batch.negative_targets.idxs, filter = filter)
    pos = loss_value(values = pos, target_type = TargetType.POSITIVE, reduce=True)
    neg = loss_value(values = neg, target_type = TargetType.NEGATIVE, reduce=True)
    return (pos + neg) / 2

def var_choices(n : int) -> List[int]:
    """
    given a number representing a variable choice, returns variables used

    Args:
        n (int): variable choice

    Returns:
        List[int]: variables used
    """
    #
    return [int(n) // 3, n % 3]

def body_pred_str(choice : int, problem : loader.Problem) -> str:
    """
    given a rule number, returns a body predicate

    Args:
        choice (int): rule number
        problem (loader.Problem): problem description

    Returns:
        str: body predicate
    """
    chosen_pred = choice // 9
    chosen_vs = choice % 9
    vs = ','.join(map(lambda v: chr(ord('A')+v), var_choices(chosen_vs)))
    return f'{problem.predicate_name[chosen_pred]}({vs})'

def clause_str(pred_name : str, choice1 : int, choice2 : int, problem : loader.Problem) -> str:
    """
    given rule numbers for body predicates, returns a clause

    Args:
        pred_name (str): name od predicate being defined
        choice1 (int): rule number for first body predicate
        choice2 (int): rule number for second body predicate
        problem (loader.Problem): problem description

    Returns:
        str: clause
    """
    return f"{pred_name.rjust(10, ' ')}(A,B) :- {body_pred_str(choice1, problem)}, {body_pred_str(choice2, problem)}."

def print_program(problem : loader.Problem, weights : torch.Tensor, split : int):
    """
    print out a program constructed with highest weights

    Args:
        problem (loader.Problem): problem description
        weights (torch.Tensor): rule weights
        split (int): how is the program split

    Raises:
        NotImplementedError: when given split outside of {0,1,2}
    """
    pred_dim = len(problem.predicate_name)
    if split == 2:
        for pred in list(problem.targets) + list(problem.invented):
            pred_name = problem.predicate_name[pred]
            for clause in range(2):
                choices = weights[pred,clause].max(-1).indices.tolist()
                print(clause_str(pred_name, choices[0], choices[1], problem))
    elif split == 1:
        for pred in list(problem.targets) + list(problem.invented):
            pred_name = problem.predicate_name[pred]
            d = pred_dim * 9
            for clause in range(2):                
                rule_no = int(weights[pred,clause].max(-1).indices.item())
                print(clause_str(pred_name, rule_no // d, rule_no % d, problem))
    elif split == 0:
        for pred in list(problem.targets) + list(problem.invented):
            pred_name = problem.predicate_name[pred]
            d = pred_dim * 9
            d2 = d * d
            rule_no = int(weights[pred].max(-1).indices.item())
            for clause_choice in (rule_no // d2, rule_no % d2):
                print(clause_str(pred_name, clause_choice // d, clause_choice % d, problem))
    else: 
        raise NotImplementedError
