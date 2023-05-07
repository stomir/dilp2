import fire #type: ignore
import dilp
import torch
from tqdm import tqdm #type: ignore
import torch
import logging
#from core import Term, Atom
from typing import *
import itertools
import numpy
import random
import os
import loader
import torcher
import sys
import traceback
from collections import defaultdict
from flipper import Flipper
import torch.nn.functional as F

def report_tensor(vals : Sequence[torch.Tensor], batch : torcher.WorldsBatch) -> torch.Tensor:
    """
    extracts inferred and expected values in a batch
    """
    target_values = torch.cat([
        torch.ones(len(batch.positive_targets), device=vals[0].device),
        torch.zeros(len(batch.negative_targets), device=vals[0].device)]).unsqueeze(1)
    
    idxs = torch.cat([batch.positive_targets.idxs, batch.negative_targets.idxs])
    other_values = [dilp.extract_targets(val, idxs).unsqueeze(1) for val in vals]

    return torch.cat([target_values] + other_values, dim=1)
    
def clip_parameters(params : Iterable[torch.nn.Parameter], min_parameter : Optional[float], max_parameter : Optional[float]) -> None:
    """
    clips parameters to given min and max values (or does nothing if given `None`)

    Args:
        params (Iterable[torch.nn.Parameter]): parameter
        min_parameter (Optional[float]): minimum value
        max_parameter (Optional[float]): maximum value
    """
    if min_parameter is not None or max_parameter is not None:
        with torch.no_grad():
            for param in params:
                if min_parameter is not None:
                    param[:] = torch.max(param, torch.as_tensor(min_parameter, device=param.device))
                if max_parameter is not None:
                    param[:] = torch.min(param, torch.as_tensor(max_parameter, device=param.device))

def main(task : str, 
        epochs : int, steps : int, 
        batch_size : float = 0.5,
        cuda : Union[int,bool] = False,
        inv : int = 0,
        debug : bool = False,
        norm : str = 'mixed',
        entropy_weight : float = 0.0,
        optim : str = 'adam', lr : float = 0.05,
        clip : Optional[float] = None,
        info : bool = False,
        entropy_enable_threshold : Optional[float] = None,
        normalize_gradients : Optional[float] = None,
        init : str = 'uniform',
        init_size : float = 10.0,        
        entropy_weight_step = 1.0,
        end_early : Optional[float] = 1e-3,
        seed : Optional[int] = None,
        validate : bool = True,
        validation_steps : Union[float,int] = 2.0,
        validate_training : bool = True,
        worlds_batch_size : int = 1,
        devices : Optional[List[int]] = None,
        entropy_gradient_ratio : Optional[float] = None,
        input : Optional[str] = None, output : Optional[str] = None,
        use_float64 : bool = False,
        use_float16 : bool = False,
        checkpoint : Optional[str] = None,
        validate_on_cpu : bool = True,
        training_worlds : Optional[int] = None,
        truth_loss : float = 0.0,
        diversity_loss : float = 0.0,
        rerandomize : float = 0.0,
        rerandomize_interval : int = 1,
        softmax_temp : Optional[float] = 1.0,
        norm_p : float = 1.0,
        target_copies : int = 0,
        split : int = 2,
        min_parameter : Optional[float] = None,
        max_parameter : Optional[float] = None,
        compile : bool = True,
        **rules_args):
    
    if info:
        logging.getLogger().setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    inv = int(inv)
    steps = int(steps)


    if use_float64:
        dtype = torch.float64
    elif use_float16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        
    if compile and inv == 0:
        logging.warning("There's a known bug where using inv==0 and torch.compile together prevents learning. Setting compile=False")
        compile = False

    if input is not None:
        input = input.format(**locals())
    
    if output is not None:
        output = output.format(**locals())

    if checkpoint is not None:
        checkpoint = checkpoint.format(**locals())
        assert input is None and output is None
        if not os.path.isfile(checkpoint):
            logging.warning(f"No file {checkpoint} found, starting from scratch")
        else:
            input = checkpoint
        output = checkpoint

    #set up random seed
    if seed is not None:
        seed = int(seed)
        torch.use_deterministic_algorithms(True) #type: ignore
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) #type: ignore

    dev = torch.device(cuda if type(cuda) == int else 0) if cuda or cuda == 0 else torch.device('cpu')

    logging.info(f'{dev=}')

    if devices is None:
        devs : Optional[List[torch.device]] = None
    else:
        devs = [torch.device(i) for i in devices]

    try:
        x = torch.zeros(size=(), device=dev).item()
    except RuntimeError as e:
        logging.error(f"device error {cuda=} {dev=}")
        raise e
    
    #load problem definition
    dirs = [d for d in os.listdir(task) if os.path.isdir(os.path.join(task, d))]
    logging.info(f'{dirs=}')
    problem = loader.load_problem(os.path.join(task, dirs[0]), invented_count=inv, target_copies=target_copies)

    #find training and validation worlds
    train_worlds = [loader.load_world(os.path.join(task, d), problem = problem, train = True) for d in dirs if d.startswith('train')]
    validation_worlds = [loader.load_world(os.path.join(task, d), problem = problem, train = False) for d in dirs if d.startswith('val')] \
                        + (train_worlds if validate_training else [])
                        
    if training_worlds is not None:
        train_worlds = train_worlds[:training_worlds]

    #split training data into batches
    worlds_batches : Sequence[torcher.WorldsBatch] = [torcher.targets_batch(problem, worlds, dev, dtype) for worlds in torcher.chunks(worlds_batch_size, train_worlds)]

    #prepare rules
    rulebook = torcher.rules(problem, dev, split = split, **rules_args)

    shape = rulebook.mask.shape

    #set up torch module
    module = dilp.DILP(
        norms=dilp.Norms.from_name(norm, p=norm_p),
        device=dev,
        devices=[torch.device(i) for i in devices] if devices is not None else None,
        rulebook=rulebook,
        init_type=init,
        init_size=init_size,
        steps=steps,
        split=split,
        softmax_temp=softmax_temp,
        problem=problem)

    module_opt = torch.compile(module) if compile else module

    params : Sequence[torch.nn.Parameter] = list(module.parameters())
    epoch : int = 0

    clip_parameters(params, min_parameter, max_parameter)

    #set up optimizer
    if optim == 'rmsprop':
        opt : torch.optim.Optimizer = torch.optim.RMSprop(params, lr=lr)
    elif optim == 'adam':
        opt = torch.optim.Adam(params, lr=lr)
    elif optim == 'sgd':
        opt = torch.optim.SGD(params, lr=lr)
    elif optim == 'flipper':
        opt = Flipper(params, lr=lr)
    else:
        assert False

    entropy_enabled = entropy_enable_threshold is None
    entropy_weight_in_use = 0.0 if entropy_enable_threshold is not None else 1.0
    
    logging.debug(f'{problem=}')

    #load weights if necessary
    if input is not None:
        dilp_sd, opt_sd, epoch, entropy_enabled, entropy_weight_in_use = torch.load(input)
        module.load_state_dict(dilp_sd)
        opt.load_state_dict(opt_sd)
        logging.info(f'loaded weights from `{input}`')
        del dilp_sd, opt_sd
        

    for _ in (tq := tqdm(range(0, int(epochs)))):
        epoch += 1
        opt.zero_grad()

        #choose which examples are going to be counted
        chosen_per_world_batch : Dict[loader.TargetType, Sequence[torch.Tensor]] = dict((ttype, 
                            [(torch.rand(len(batch.targets(ttype)), device=dev) <= batch_size) for batch in worlds_batches]) for ttype in loader.TargetType)
        chosen_per_ttype : Dict[loader.TargetType, int] = dict((ttype, sum(int(c.sum().item()) for c in chosen_per_world_batch[ttype])) for ttype in loader.TargetType)

        all_worlds_sizes = dict((t, sum(len(b.targets(t)) for b in worlds_batches)) for t in loader.TargetType)

        loss_sum = 0.0

        try:
            target_losses : List[float] = []
            for i, batch in enumerate(worlds_batches):

                #compute inference
                vals = module_opt(batch.base_val) #type: ignore

                ls = torch.as_tensor(0.0, device=dev)
                one_target_loss = 0.0
                #get positive and negative loss
                for target_type in loader.TargetType:
                    targets : torch.Tensor = batch.targets(target_type).idxs.to(dev, non_blocking=False)
                    preds = dilp.extract_targets(vals, targets)
                    loss = dilp.loss_value(preds, target_type, reduce=False)

                    one_target_loss += loss.detach().mean().item() / 2

                    chosen = chosen_per_world_batch[target_type][i]
                    if chosen.sum() != 0:
                        loss = (loss.where(chosen, torch.zeros(size=(), device=loss.device)).sum()) / chosen.sum()
                        loss = loss * int(chosen.sum().item()) / chosen_per_ttype[target_type] / 2
                    else:
                        loss = torch.zeros(size=(), device=loss.device)

                    ls = ls + loss

                #store total loss (not limited to choices for this batch)
                target_losses.append(one_target_loss)

                assert ls >= 0

                #optionally apply experimental auxiliary loss
                if truth_loss != 0.0:
                    ls = ls + vals.mean(0).sum(0).mean(0).mean(0) * truth_loss

                #optionally apply experimental auxiliary loss
                if diversity_loss != 0.0:
                    print(f"{vals.shape=}")
                    ls = ls + (vals.unsqueeze(2) - vals.unsqueeze(1)).square().mean(0).sum(0).mean(0).mean(0).mean(0) * diversity_loss
                
                #backpropagate
                if ls != 0.0:
                    ls.backward()
                loss_sum += ls.item()

                avg_vals = vals.mean().item()
                del loss, vals, targets, preds, ls
                torch.cuda.empty_cache()
            
            #optionally apply experimental auxiliary loss
            if normalize_gradients is not None:
                with torch.no_grad():
                    for fuzzy in params:
                        if fuzzy.grad is not None:
                            logging.info(f"{fuzzy.grad.norm(2)=}")
                            fuzzy.grad[:] = torch.nn.functional.normalize(fuzzy.grad, dim=-1)
                            fuzzy.grad *= normalize_gradients
                
            if entropy_enable_threshold is not None and loss_sum < entropy_enable_threshold:
                entropy_enabled = True
            
            entropy_loss : torch.Tensor = norm_loss(dilp.mask(module.weights, rulebook))
            entropy_loss = dilp.mask(entropy_loss, rulebook)
            actual_entropy = entropy_loss.mean()
            if entropy_enabled:
                if entropy_gradient_ratio is not None:
                    entropy_loss = entropy_loss * entropy_gradient_ratio * module.weights.norm(p=2, dim=-1, keepdim=True)
                entropy_loss = entropy_loss.mean()
                if entropy_weight_in_use < 1.0 and entropy_enable_threshold is not None and loss_sum < entropy_enable_threshold:
                    entropy_weight_in_use += entropy_weight_step
                entropy_loss = entropy_loss * entropy_weight_in_use * entropy_weight
                entropy_loss.backward()

            #compute total loss
            target_loss = sum(target_losses) / len(target_losses)
            if end_early is not None and target_loss < end_early:
                break

            #clip gradients
            if clip is not None:
                torch.nn.utils.clip_grad_value_([module.weights], clip)

            #perform optimization step
            opt.step()

            #optionally clip weights
            clip_parameters(params, min_parameter, max_parameter)

        except AssertionError as e:
            weights_file : str = output + "_faulty" if output is not None else "faulty_weights"
            logging.error(f"assertion during backprop, saving weights to {weights_file}\n{traceback.format_exc()}")
            torch.save(module.state_dict(), weights_file)
            raise e
        
        report = {'entropy' : actual_entropy.item(), 'b_loss' : loss_sum, 
                'loss' : target_loss,
                }
        tq.set_postfix(**report) #type: ignore

    #print final program
    dilp.print_program(problem, dilp.mask(module.weights, rulebook), split=split)

    if output is not None:
        torch.save((module.state_dict(), opt.state_dict(), epoch, entropy_enabled, entropy_weight_in_use), output)
        logging.info(f"saved weights to `{output}`")
    
    #perform validation
    if validate:
        with torch.no_grad():
            last_target = target_loss
            last_entropy = actual_entropy.item()
            
            #optionally switch to computing on CPU
            if validate_on_cpu:
                dev = torch.device('cpu')
                devs = None
                module = module.to(dev)
                module.rulebook = module.rulebook.to(dev)
            
            #choose a target copy to use as an answer (optionally)
            if target_copies == 0:
                chosen_targets : Set[int] = problem.targets
            else:
                losses : DefaultDict[int, float] = defaultdict(lambda: 0.0)
                
                for world in train_worlds:
                    vals = torcher.base_val(problem, [world], dtype=dtype).to(dev)
                    vals = module(base_val = vals)
                    batch = torcher.targets_batch(problem, [world], dev, dtype=dtype)
                    for original, copies in problem.target_copies.items():
                        for pred in copies + [original]:
                            loss = dilp.loss(vals = vals, batch = batch.filter(lambda t: t[1] == pred))
                            losses[pred] += loss.item()
                
                chosen_targets = set()        
                for original, copies in problem.target_copies.items():
                    _, choice = min((losses[pred], pred) for pred in copies + [original])
                    logging.info(f'{problem.predicate_name[original]}: chosen copy was {problem.predicate_name[choice]}')
                    chosen_targets.add(choice)
            
            #compute validation
            valid_worlds = 0
            fuzzily_valid_worlds = 0
            valid_tr_worlds = 0
            fuzzily_valid_tr_worlds = 0
            rulebook = rulebook.to(dev, non_blocking=True)
            if type(validation_steps) is float:
                module.steps = int(steps * validation_steps)
            else:
                module.steps = int(validation_steps)
            for i, world in enumerate(validation_worlds):
                batch = torcher.targets_batch(problem, [world], dev, dtype=dtype)
                batch = batch.filter(lambda target: target[1] in chosen_targets)
                base_val = batch.base_val.to(dev)
                fuzzy_vals = module(base_val)
                logging.info(f"{fuzzy_vals.mean()=}")

                crisp_vals = module(base_val, crisp=True)

                report_t = report_tensor([fuzzy_vals, crisp_vals], batch)
                fuzzy_report = report_t[:,1]
                crisp_report = report_t[:,2]
                target_values = report_t[:,0]
                fuzzy_acc = (fuzzy_report.round() == target_values).float().mean().item()
                crisp_acc = (crisp_report == target_values).float().mean().item()
                logging.info(f'world {i} {world.dir=} {fuzzy_acc=} {crisp_acc=}\n{report_t.cpu().numpy()}')
                if crisp_acc == 1.0:
                    valid_worlds += 1
                    if world.train:
                        valid_tr_worlds += 1
                if fuzzy_acc == 1.0:
                    fuzzily_valid_worlds += 1
                    if world.train:
                        fuzzily_valid_tr_worlds += 1
            
            result = '     OK      ' if valid_worlds == len(validation_worlds) else \
                    '    FUZZY    ' if fuzzily_valid_worlds == len(validation_worlds) else \
                    '   OVERFIT   ' if valid_tr_worlds == len(train_worlds) else \
                    'FUZZY OVERFIT' if fuzzily_valid_tr_worlds == len(train_worlds) else \
                    '    FAIL     '
            print(f'result: {result} {valid_worlds=} {fuzzily_valid_worlds=} ' +
                    f' {last_target=} {last_entropy=} {epoch=}')

def norm_loss(weights : torch.Tensor) -> torch.Tensor:
    x = weights
    x = (x.softmax(-1) * x.log_softmax(-1))

    return -x.sum()


if __name__ == "__main__":
    fire.Fire(main)
