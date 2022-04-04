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
import sys
import traceback
from torch.utils.tensorboard import SummaryWriter

def mask(t : torch.Tensor, rulebook : dilp.Rulebook) -> torch.Tensor:
    return t.where(rulebook.mask, torch.zeros(size=(),device=t.device))

def masked_softmax(t : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
    t = t.where(mask, torch.as_tensor(-float('inf'), device=t.device)).softmax(-1)
    t = t.where(t.isnan().logical_not(), torch.as_tensor(0.0, device=t.device)) #type: ignore
    return t

def report_tensor(vals : Sequence[torch.Tensor], batch : torcher.WorldsBatch) -> torch.Tensor:
    target_values = torch.cat([
        torch.ones(len(batch.positive_targets)),
        torch.zeros(len(batch.negative_targets))]).unsqueeze(1)
    
    idxs = torch.cat([batch.positive_targets.idxs, batch.negative_targets.idxs])
    other_values = [dilp.extract_targets(val, idxs).unsqueeze(1) for val in vals]

    return torch.cat([target_values] + other_values, dim=1)
    

def main(task : str, 
        epochs : int, steps : int, 
        batch_size : Optional[int],
        cuda : Union[int,bool] = False, inv : int = 0,
        debug : bool = False, norm : str = 'mixed',
        entropy_weight : float = 1.0,
        optim : str = 'adam', lr : float = 0.05, clip : Optional[float] = None,
        init_rand : float = 10,
        info : bool = False,
        entropy_enable_threshold : Optional[float] = 1e-2,
        normalize_gradients : Optional[float] = None,
        init : str = 'uniform',
        entropy_weight_step = 1,
        end_early : Optional[float] = 1e-3,
        seed : Optional[int] = None,
        validate : bool = True,
        validation_steps : Optional[int] = None,
        validate_training : bool = True,
        worlds_batch_size : int = 1,
        devices : Optional[List[int]] = None,
        entropy_gradient_ratio : Optional[float] = None,
        cut_down_rules : Union[int,float,None] = None,
        input : Optional[str] = None, output : Optional[str] = None,
        tensorboard : Optional[str] = None,
        use_float64 : bool = False,
        **rules_args):
    if info:
        logging.getLogger().setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if use_float64:
        torch.set_default_tensor_type(torch.DoubleTensor)

    if seed is not None:
        seed = int(seed)
        torch.use_deterministic_algorithms(True) #type: ignore
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) #type: ignore

    dilp.set_norm(norm)
    dev = torch.device(cuda if type(cuda) == int else 0) if cuda != False else torch.device('cpu')

    logging.info(f'{dev=}')

    if tensorboard is not None:
        tb : Optional[SummaryWriter] = SummaryWriter(log_dir=tensorboard, comment=' '.join(sys.argv))
    else:
        tb = None

    if devices is None:
        devs : Optional[List[torch.device]] = None
    else:
        devs = [torch.device(i) for i in devices]

    if  inv<0:
        raise DataError('The number of invented predicates must be >= 0')

    try:
        x = torch.zeros(size=(), device=dev).item()
    except RuntimeError as e:
        logging.error(f"device error {cuda=} {dev=}")
        raise e
    

    dirs = [d for d in os.listdir(task) if os.path.isdir(os.path.join(task, d))]
    logging.info(f'{dirs=}')
    problem = loader.load_problem(os.path.join(task, dirs[0]), invented_count=inv)

    train_worlds = [loader.load_world(os.path.join(task, d), problem = problem) for d in dirs if d.startswith('train')]
    validation_worlds = [loader.load_world(os.path.join(task, d), problem = problem) for d in dirs if d.startswith('val')] \
                        + (train_worlds if validate_training else [])

    worlds_batches : Sequence[torcher.WorldsBatch] = [torcher.targets_batch(problem, worlds, dev) for worlds in torcher.chunks(worlds_batch_size, train_worlds)]
    choices : Sequence[numpy.ndarray] = [numpy.concatenate([numpy.repeat(i, len(batch.targets(target_type).idxs)) for i, batch in enumerate(worlds_batches)]) for target_type in loader.TargetType]

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    rulebook = torcher.rules(problem, dev, **rules_args)

    if cut_down_rules is not None:
        if type(cut_down_rules) == int:
            rulebook = dilp.cut_down_rules(rulebook, cut_down_rules)
        elif type(cut_down_rules) == float:
            rulebook = dilp.cut_down_rules(rulebook, int(rulebook.body_predicates.shape[-1] * cut_down_rules))
        else:
            assert False, "cut_down_rules is not int or float"

    shape = rulebook.body_predicates.shape

    weights : torch.nn.Parameter = torch.nn.Parameter(torch.normal(mean=torch.zeros(size=[shape[0], 2, 2, shape[3]], device=dev), std=init_rand)) \
        if init == 'normal' else torch.nn.Parameter(torch.rand([shape[0], 2, 2, shape[3]], device=dev) * init_rand)

    if input is not None:
        with torch.load(input) as w:
            weights[:] = w.to(dev)
            logging.info(f'loaded weights from {input}')

    #opt = torch.optim.SGD([weights], lr=1e-2)
    if optim == 'rmsprop':
        opt : torch.optim.Optimizer = torch.optim.RMSprop([weights], lr=lr)
    elif optim == 'adam':
        opt = torch.optim.Adam([weights], lr=lr)
    elif optim == 'sgd':
        opt = torch.optim.SGD([weights], lr=lr)
    else:
        assert False
        
    entropy_enabled = entropy_enable_threshold is None
    entropy_weight_in_use = 0.0 if entropy_enable_threshold is not None else 1.0

    for epoch in (tq := tqdm(range(0, int(epochs)))):
        opt.zero_grad()

        chosen_worlds_batches = [numpy.random.choice(choices_of_type, replace=False, size=batch_size) for choices_of_type in choices]

        all_worlds_sizes = dict((t, sum(len(b.targets(t)) for b in worlds_batches)) for t in loader.TargetType)

        loss_sum = 0.0

        try:

            for i, batch in enumerate(worlds_batches):
                ws = masked_softmax(weights, rulebook.mask)

                assert (ws < 0).sum() == 0 and (ws > 1).sum() == 0, f"{ws=} {i=}"

                vals = dilp.infer(base_val = batch.base_val, rulebook = rulebook, 
                            weights = ws, steps=steps, devices = devs)

                assert (vals < 0).sum() == 0 and (vals > 1).sum() == 0, f"{(vals < 0).sum()=} {(vals > 1).sum()=} {i=} {steps=} {devices=} {vals.max()=}"

                if batch_size is not None:
                    num_to_use_in_this_batch : Sequence[int] = [(chosen_worlds_batches_of_type == i).sum() for chosen_worlds_batches_of_type in chosen_worlds_batches]

                    if sum(num_to_use_in_this_batch) == 0:
                        logging.debug(f'skipped worlds batch {i} as nothing was chosen')
                        continue

                    to_use_in_this_batch : Sequence[numpy.ndarray] = [numpy.random.choice(numpy.arange(len(batch.targets(target_type))), replace=False, size=to_choose)
                                for target_type, to_choose in zip(loader.TargetType, num_to_use_in_this_batch)]

                    ls = torch.as_tensor(0.0, device=dev)
                    for target_type, to_use_in_this_batch_of_type in zip(loader.TargetType, to_use_in_this_batch):
                        if len(to_use_in_this_batch_of_type) == 0:
                            continue
                        targets = batch.targets(target_type).idxs[torch.from_numpy(to_use_in_this_batch_of_type).to(dev, non_blocking=False)]
                        preds = dilp.extract_targets(vals, targets)
                        loss = dilp.loss(preds, target_type)
                        loss = loss * (len(to_use_in_this_batch_of_type) / batch_size / 2)

                        assert loss >= 0, f"{target_type=} {loss=} {preds=} {vals=}"

                        ls = ls + loss
                else:

                    ls = torch.as_tensor(0.0, device=dev)
                    for target_type in loader.TargetType:
                        targets = batch.targets(target_type).idxs.to(dev, non_blocking=False)
                        preds = dilp.extract_targets(vals, targets)
                        loss = dilp.loss(preds, target_type)

                        assert loss >= -1e-5, f"{target_type=} {loss=} {preds=}"

                        ls = ls + loss

                    ls = ls * len(batch.targets(target_type)) / all_worlds_sizes[target_type] / 2

                ls.backward()
                assert ls >= 0
                loss_sum += ls.item()

                del loss, vals, targets, preds, ls
                torch.cuda.empty_cache()
            
            if normalize_gradients is not None:
                with torch.no_grad():
                    for fuzzy in weights:
                        #w /= w.sum(-1, keepdim=True) * w.sign()
                        fuzzy[:] = torch.nn.functional.normalize(fuzzy, dim=-1)
                        fuzzy *= normalize_gradients
                
            if entropy_enable_threshold is not None and loss_sum < entropy_enable_threshold:
                entropy_enabled = True
            
            entropy_loss : torch.Tensor = norm_loss(mask(weights, rulebook))
            entropy_loss = mask(entropy_loss, rulebook)
            actual_entropy = entropy_loss.mean()
            if entropy_enabled:
                if entropy_gradient_ratio is not None:
                    entropy_loss = entropy_loss * entropy_gradient_ratio * weights.norm(p=2, dim=-1, keepdim=True)
                entropy_loss = entropy_loss.mean()
                if entropy_weight_in_use < 1.0 and entropy_enable_threshold is not None and loss_sum < entropy_enable_threshold:
                    entropy_weight_in_use += entropy_weight_step
                entropy_loss = entropy_loss * entropy_weight_in_use * entropy_weight
                entropy_loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(weights, clip)

            opt.step()
            #adjust_weights(weights)
        except AssertionError as e:
            weights_file : str = output + "_faulty" if output is not None else "faulty_weights"
            logging.error(f"assertion during backprop, saving weights to {weights_file}\n{traceback.format_exc()}")
            torch.save(weights.detach(), weights_file)
            raise e

        tq.set_postfix(entropy = actual_entropy.item(), batch_loss = loss_sum, entropy_weight=entropy_weight_in_use * entropy_weight)
        if tb is not None:
            tb.add_scalars("train", 
                {'entropy' : actual_entropy.item(), 'batch_loss' : loss_sum, 
                'entropy_weight' : entropy_weight_in_use * entropy_weight},
                global_step=epoch)

        if end_early is not None and loss_sum < end_early:
            break

    dilp.print_program(rulebook, mask(weights, rulebook), torcher.rev_dict(problem.predicates))
    
    if validate:
        last_target = loss_sum
        last_entropy = actual_entropy.item()
        with torch.no_grad():
            total_loss = 0.0
            total_fuzzy = 0.0
            valid_worlds = 0
            fuzzily_valid_worlds = 0
            dev = torch.device('cpu')
            rulebook = rulebook.to(dev, non_blocking=False)
            fuzzy = weights.detach().to(dev, non_blocking=False)
            crisp = mask(torch.nn.functional.one_hot(fuzzy.max(-1)[1], fuzzy.shape[-1]).float(), rulebook)
            if validation_steps is None:
                validation_steps = steps * 2
            for i, world in enumerate(validation_worlds):
                base_val = torcher.base_val(problem, [world])
                batch = torcher.targets_batch(problem, [world], dev)
                fuzzy_vals = dilp.infer(base_val, rulebook, weights = masked_softmax(fuzzy, rulebook.mask), steps=validation_steps)
                fuzzy_loss : torch.Tensor = sum((dilp.loss(dilp.extract_targets(fuzzy_vals, batch.targets(target_type).idxs), target_type) for target_type in loader.TargetType), start=torch.as_tensor(0.0))

                crisp_vals = dilp.infer(base_val, rulebook, weights = crisp, steps=validation_steps)
                crisp_loss : torch.Tensor = sum((dilp.loss(dilp.extract_targets(crisp_vals, batch.targets(target_type).idxs), target_type) for target_type in loader.TargetType), start=torch.as_tensor(0.0))

                report = report_tensor([fuzzy_vals, crisp_vals], batch)
                fuzzy_report = report[:,1]
                crisp_report = report[:,2]
                target_values = report[:,0]
                fuzzy_acc = (fuzzy_report.round() == target_values).float().mean().item()
                crisp_acc = (crisp_report == target_values).float().mean().item()
                logging.info(f'world {i} {fuzzy_acc=} {crisp_acc=}\n{report.numpy()}')
                total_loss += crisp_loss.item()
                total_fuzzy += fuzzy_loss.item()
                if crisp_acc == 1.0:
                    valid_worlds += 1
                if fuzzy_acc == 1.0:
                    fuzzily_valid_worlds += 1
            
            #valid_worlds /= len(validation_worlds)
            #fuzzily_valid_worlds /= len(validation_worlds)
            result = ' OK ' if valid_worlds == len(validation_worlds) else \
                    'FUZZ' if fuzzily_valid_worlds == len(validation_worlds) else 'FAIL'
            print(f'result: {result} {valid_worlds=} {fuzzily_valid_worlds=} {total_loss=} ' +
                      f'{total_fuzzy=} {last_target=} {last_entropy=} {epoch=}')
    
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
    #logsoftmax = x.log_softmax(-1)
    #softmax = x.softmax(-1)
    x = (x.softmax(-1) * x.log_softmax(-1))
    #x = (x * x.log())
    return -x.sum()

if __name__ == "__main__":
    fire.Fire(main)
