from sqlite3 import DataError
import fire #type: ignore
import dilp
import torch
from tqdm import tqdm #type: ignore
import torch
import logging
#from core import Term, Atom
from typing import *
import GPUtil #type: ignore
import itertools
import numpy
import random
import os
import loader
import torcher

def mask(t : torch.Tensor, rulebook : dilp.Rulebook) -> torch.Tensor:
    return t.where(rulebook.mask, torch.zeros(size=(),device=t.device))

def masked_softmax(t : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
    t = t.where(mask, torch.as_tensor(-float('inf'), device=t.device)).softmax(-1)
    t = t.where(t.isnan().logical_not(), torch.as_tensor(0.0, device=t.device)) #type: ignore
    return t

def main(task : str, epochs : int = 100, steps : int = 1, cuda : Optional[Union[int,bool]] = None, inv : int = 0,
        debug : bool = False, norm : str = 'mixed', norm_weight : float = 1.0,
        optim : str = 'adam', lr : float = 0.05, clip : Optional[float] = None,
        init_rand : float = 10,
        info : bool = False,
        normalize_threshold : Optional[float] = 1e-2,
        batch_size : Optional[int] = None,
        normalize_gradients : Optional[float] = None,
        init : str = 'uniform',
        entropy_weight_step = 1,
        end_early : Optional[float] = 1e-3,
        seed : Optional[int] = None, dropout : float = 0,
        validate : bool = True,
        validation_steps : Optional[int] = None,
        validate_training : bool = True,
        input : Optional[str] = None, output : Optional[str] = None,
        **rules_args):
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
    dev = torch.device(cuda if type(cuda) == int else 0) if cuda is not None else torch.device('cpu')

    if  inv<0:
        raise DataError('The number of invented predicates must be >= 0')

    

    dirs = [d for d in os.listdir(task) if os.path.isdir(os.path.join(task, d))]
    problem = loader.load_problem(os.path.join(task, dirs[0]), invented_count=inv)

    train_worlds = [loader.load_world(os.path.join(task, d), problem = problem) for d in dirs if d.startswith('train')]
    val_worlds = [loader.load_world(os.path.join(task, d), problem = problem) for d in dirs if d.startswith('val')] + 
                        (train_worlds if validate_training else [])

    base_val = torcher.base_val(problem, worlds = train_worlds).to(dev)
    positive_targets = torcher.targets(train_worlds, positive = True).to(dev)
    negative_targets = torcher.targets(train_worlds, positive = False).to(dev)
    targets, target_values = torcher.targets_tuple(train_worlds, device = dev)

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    rulebook = torcher.rules(problem, dev, **rules_args)
    shape = rulebook.body_predicates.shape

    weights : torch.nn.Parameter = torch.nn.Parameter(torch.normal(mean=torch.zeros(size=[shape[0], 2, 2, shape[3]], device=dev), std=init_rand)) \
        if init == 'normal' else torch.nn.Parameter(torch.rand([shape[0], 2, 2, shape[3]], device=dev) * init_rand)

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
            moved = masked_softmax(weights, rulebook.mask)
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
                for fuzzy in weights:
                    #w /= w.sum(-1, keepdim=True) * w.sign()
                    fuzzy[:] = torch.nn.functional.normalize(fuzzy, dim=-1)
                    fuzzy *= normalize_gradients

        opt.step()
        #adjust_weights(weights)

        tq.set_postfix(target_loss = report_loss.item(), entropy_loss = entropy_loss.item(), batch_loss = target_loss.item(), entropy_weight=entropy_weight * norm_weight)

        logging.info(f"target loss: {report_loss.item()} entropy loss: {entropy_loss.item()}")

        if end_early is not None and report_loss.item() < end_early:
            break

    dilp.print_program(rulebook, mask(weights, rulebook), torcher.rev_dict(problem.predicates))
    
    if validate:
        with torch.no_grad():
            total_loss = 0.0
            dev = torch.device('cpu')
            rulebook = rulebook.to(dev)
            fuzzy = weights.detach().to(dev)
            crisp = mask(torch.nn.functional.one_hot(fuzzy.max(-1)[1], fuzzy.shape[-1]).float(), rulebook)
            if validation_steps is None:
                validation_steps = steps * 2
            for world in val_worlds:
                base_val = torcher.base_val(problem, [world])
                targets, target_values = torcher.targets_tuple([world], device = dev)
                fuzzy_loss, fuzzy_report = dilp.loss(base_val, rulebook=rulebook, weights = masked_softmax(fuzzy, rulebook.mask), targets=targets, target_values=target_values, steps=validation_steps)
                crisp_loss, crisp_report = dilp.loss(base_val, rulebook=rulebook, weights = crisp, targets=targets, target_values=target_values, steps=validation_steps)
                report = torch.cat([target_values.unsqueeze(1), fuzzy_report.unsqueeze(1), crisp_report.unsqueeze(1)], dim=1).detach().cpu().numpy()
                print(f'fuzzy_loss: {fuzzy_loss.mean().item():.5f} crisp_loss: {crisp_loss.mean().item():.5f}:\n', report)
                total_loss += crisp_loss.sum().item()
            
            if total_loss == 0.0:
                print('result: OK')
            else:
                print('result: FAIL')
    
    if output is not None:
        torch.save(weights.detach().cpu(), output)

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
