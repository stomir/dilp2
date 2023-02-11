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
import plot
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from flipper import Flipper
import torch.nn.functional as F

def mask(t : torch.Tensor, rulebook : dilp.Rulebook) -> torch.Tensor:
    return t.where(rulebook.mask, torch.zeros(size=(),device=t.device))

def masked_softmax(t : torch.Tensor, mask : torch.Tensor, temp : Optional[float]) -> torch.Tensor:
    if temp is None:
        t = t.where(mask, torch.as_tensor(0, device=t.device))
    else:
        t = t.where(mask, torch.as_tensor(-float('inf'), device=t.device)).softmax(-1)
        t = t.where(t.isnan().logical_not(), torch.as_tensor(0.0, device=t.device)) #type: ignore
    return t

def report_tensor(vals : Sequence[torch.Tensor], batch : torcher.WorldsBatch) -> torch.Tensor:
    target_values = torch.cat([
        torch.ones(len(batch.positive_targets), device=vals[0].device),
        torch.zeros(len(batch.negative_targets), device=vals[0].device)]).unsqueeze(1)
    
    idxs = torch.cat([batch.positive_targets.idxs, batch.negative_targets.idxs])
    other_values = [dilp.extract_targets(val, idxs).unsqueeze(1) for val in vals]

    return torch.cat([target_values] + other_values, dim=1)

def random_init(init : str, device : torch.device, shape : List[int], init_size : float, dtype) -> torch.Tensor:
    if init == 'normal':
        return torch.normal(mean=torch.zeros(size=shape, device=device, dtype=dtype), std=init_size)
    elif init == 'uniform':
        return torch.rand(size=shape, device=device, dtype=dtype) * init_size
    elif init == 'discrete':
        return F.one_hot(torch.randint(low=0, high=shape[-1], size=shape[:-1], device=device), num_classes = shape[3]).float()
    else:
        raise RuntimeError(f'unknown init: {init}')
    

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
        init : str = 'normal',
        init_size : float = 1.0,        
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
        tensorboard : Optional[str] = None,
        use_float64 : bool = False,
        use_float16 : bool = False,
        checkpoint : Optional[str] = None,
        validate_on_cpu : bool = True,
        training_worlds : Optional[int] = None,
        truth_loss : float = 0.0,
        diversity_loss : float = 0.0,
        rerandomize : float = 0.0,
        rerandomize_interval : int = 1,
        plot_output : Optional[str] = None,
        plot_interval : int = 100,
        softmax_temp : Optional[float] = 1.0,
        norm_p : float = 1.0,
        true_init_bias : float = 0.0,
        target_copies : int = 0,
        split : int = 2,
        min_parameter : Optional[float] = None,
        max_parameter : Optional[float] = None,
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

    if seed is not None:
        seed = int(seed)
        torch.use_deterministic_algorithms(True) #type: ignore
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) #type: ignore

    dilp.set_norm(norm, p = norm_p)
    dev = torch.device(cuda if type(cuda) == int else 0) if cuda else torch.device('cpu')

    logging.info(f'{dev=}')

    if tensorboard is not None:
        tb : Optional[SummaryWriter] = SummaryWriter(log_dir=tensorboard, comment=' '.join(sys.argv))
    else:
        tb = None

    if devices is None:
        devs : Optional[List[torch.device]] = None
    else:
        devs = [torch.device(i) for i in devices]

    try:
        x = torch.zeros(size=(), device=dev).item()
    except RuntimeError as e:
        logging.error(f"device error {cuda=} {dev=}")
        raise e
    

    dirs = [d for d in os.listdir(task) if os.path.isdir(os.path.join(task, d))]
    logging.info(f'{dirs=}')
    problem = loader.load_problem(os.path.join(task, dirs[0]), invented_count=inv, target_copies=target_copies)

    train_worlds = [loader.load_world(os.path.join(task, d), problem = problem, train = True) for d in dirs if d.startswith('train')]
    validation_worlds = [loader.load_world(os.path.join(task, d), problem = problem, train = False) for d in dirs if d.startswith('val')] \
                        + (train_worlds if validate_training else [])
                        
    if training_worlds is not None:
        train_worlds = train_worlds[:training_worlds]

    worlds_batches : Sequence[torcher.WorldsBatch] = [torcher.targets_batch(problem, worlds, dev, dtype) for worlds in torcher.chunks(worlds_batch_size, train_worlds)]
    choices : Sequence[numpy.ndarray] = [numpy.concatenate([numpy.repeat(i, len(batch.targets(target_type).idxs)) for i, batch in enumerate(worlds_batches)]) for target_type in loader.TargetType]

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    rulebook = torcher.rules(problem, dev, split = split, **rules_args)

    shape = rulebook.mask.shape

    weights : torch.nn.Parameter = torch.nn.Parameter(random_init(init, device = dev, shape = list(shape), init_size = init_size, dtype = dtype))
    params : Sequence[torch.nn.Parameter] = [weights]
    epoch : int = 0

    if true_init_bias != 0.0:
        with torch.no_grad():
            weights[:,:,:,9*problem.predicate_number['$true']] += true_init_bias

    #opt = torch.optim.SGD([weights], lr=1e-2)
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

    if input is not None:
        w, opt_sd, epoch, entropy_enabled, entropy_weight_in_use = torch.load(input)
        with torch.no_grad():
            weights[:] = w.to(dev)
        opt.load_state_dict(opt_sd)
        logging.info(f'loaded weights from `{input}`')
        del w, opt_sd
        

    for _ in (tq := tqdm(range(0, int(epochs)))):
        epoch += 1
        opt.zero_grad()

        chosen_per_world_batch : Dict[loader.TargetType, Sequence[torch.Tensor]] = dict((ttype, 
                            [(torch.rand(len(batch.targets(ttype)), device=dev) <= batch_size) for batch in worlds_batches]) for ttype in loader.TargetType)
        chosen_per_ttype : Dict[loader.TargetType, int] = dict((ttype, sum(int(c.sum().item()) for c in chosen_per_world_batch[ttype])) for ttype in loader.TargetType)

        all_worlds_sizes = dict((t, sum(len(b.targets(t)) for b in worlds_batches)) for t in loader.TargetType)

        loss_sum = 0.0

        try:
            target_losses : List[float] = []
            for i, batch in enumerate(worlds_batches):
                ws = masked_softmax(weights, rulebook.mask, temp = softmax_temp)

                vals = dilp.infer(base_val = batch.base_val, rulebook = rulebook, problem = problem,
                            weights = ws, steps=steps, devices = devs, split=split)

                #assert (vals < 0).sum() == 0 and (vals > 1).sum() == 0, f"{(vals < 0).sum()=} {(vals > 1).sum()=} {i=} {steps=} {devices=} {vals.max()=}"

                ls = torch.as_tensor(0.0, device=dev)
                one_target_loss = 0.0
                for target_type in loader.TargetType:
                    targets : torch.Tensor = batch.targets(target_type).idxs.to(dev, non_blocking=False)
                    preds = dilp.extract_targets(vals, targets)
                    loss = dilp.loss_value(preds, target_type, reduce=False)

                    one_target_loss += loss.detach().mean().item() / 2

                    if type(batch_size) is float:
                        chosen = chosen_per_world_batch[target_type][i]
                        #chosen = torch.from_numpy(chosen_here).to(dev, non_blocking=False)
                        if chosen.sum() != 0:
                            loss = (loss.where(chosen, torch.zeros(size=(), device=loss.device)).sum()) / chosen.sum()
                            loss = loss * int(chosen.sum().item()) / chosen_per_ttype[target_type] / 2
                        else:
                            loss = torch.zeros(size=(), device=loss.device)
                    else:
                        loss = loss.mean()
                        loss = loss * len(batch.targets(target_type)) / all_worlds_sizes[target_type] / 2

                    assert loss >= 0, f"{target_type=} {loss=} {preds=}"

                    ls = ls + loss

                #ls = ls 
                target_losses.append(one_target_loss)

                assert ls >= 0

                if truth_loss != 0.0:
                    ls = ls + vals.mean(0).sum(0).mean(0).mean(0) * truth_loss

                if diversity_loss != 0.0:
                    print(f"{vals.shape=}")
                    ls = ls + (vals.unsqueeze(2) - vals.unsqueeze(1)).square().mean(0).sum(0).mean(0).mean(0).mean(0) * diversity_loss
                
                if ls != 0.0:
                    ls.backward()
                loss_sum += ls.item()

                avg_vals = vals.mean().item()
                del loss, vals, targets, preds, ls
                torch.cuda.empty_cache()
            
            if normalize_gradients is not None:
                with torch.no_grad():
                    for fuzzy in params:
                        #w /= w.sum(-1, keepdim=True) * w.sign()
                        if fuzzy.grad is not None:
                            logging.info(f"{fuzzy.grad.norm(2)=}")
                            fuzzy.grad[:] = torch.nn.functional.normalize(fuzzy.grad, dim=-1)
                            fuzzy.grad *= normalize_gradients
                
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

            target_loss = sum(target_losses) / len(target_losses)
            if end_early is not None and target_loss < end_early:
                break

            if clip is not None:
                torch.nn.utils.clip_grad_value_([weights], clip)

            opt.step()

            if min_parameter is not None or max_parameter is not None:
                with torch.no_grad():
                    for param in params:
                        if min_parameter is not None:
                            param[:] = torch.max(param, torch.as_tensor(min_parameter, device=param.device))
                        if max_parameter is not None:
                            param[:] = torch.min(param, torch.as_tensor(max_parameter, device=param.device))

            # if rerandomize != 0 and (epoch-1) % int(rerandomize_interval) == 0:
            #     with torch.no_grad():
            #         weights[:] = weights * (1 - rerandomize) + random_init(init, dev, shape, init_size, dtype) * rerandomize

            if plot_output is not None and (epoch-1) % int(plot_interval) == 0:
                plot.weights_plot(weights, outdir=plot_output, epoch = epoch)

            #adjust_weights(weights)
        except AssertionError as e:
            weights_file : str = output + "_faulty" if output is not None else "faulty_weights"
            logging.error(f"assertion during backprop, saving weights to {weights_file}\n{traceback.format_exc()}")
            torch.save(weights.detach(), weights_file)
            raise e
        
        report = {'entropy' : actual_entropy.item(), 'batch_loss' : loss_sum, 
                'target_loss' : target_loss,
                #'avg_val' : avg_vals,
                #'entropy_weight' : entropy_weight_in_use * entropy_weight,
                }
        tq.set_postfix(**report)
        if tb is not None:
            tb.add_scalars("train", 
                report,
                global_step=epoch)

    dilp.print_program(problem, mask(weights, rulebook))

    if output is not None:
        torch.save((weights.detach().cpu(), opt.state_dict(), epoch, entropy_enabled, entropy_weight_in_use), output)
        logging.info(f"saved weights to `{output}`")
    
    if validate:
        with torch.no_grad():
            last_target = target_loss
            last_entropy = actual_entropy.item()
            
            if validate_on_cpu:
                dev = torch.device('cpu')
                devs = None
                
            ws = masked_softmax(weights.to(dev), rulebook.mask.to(dev), temp = softmax_temp)
            
            #chosing from target copies
            if target_copies == 0:
                chosen_targets : Set[int] = problem.targets
            else:
                losses : DefaultDict[int, float] = defaultdict(lambda: 0.0)
                
                for world in train_worlds:
                    vals = torcher.base_val(problem, [world], dtype=dtype).to(dev)
                    vals = dilp.infer(base_val = vals, rulebook = rulebook, problem = problem,
                            weights = ws, steps=steps,split=split)
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
            
            valid_worlds = 0
            fuzzily_valid_worlds = 0
            valid_tr_worlds = 0
            fuzzily_valid_tr_worlds = 0
            rulebook = rulebook.to(dev, non_blocking=True)
            fuzzy_p : torch.Tensor = weights.detach().to(dev, non_blocking=True)
            crisp = mask(torch.nn.functional.one_hot(fuzzy_p.max(-1)[1], fuzzy_p.shape[-1]).to(dev).float(), rulebook)
            if type(validation_steps) is float:
                val_steps = int(steps * validation_steps)
            else:
                val_steps = int(validation_steps)
            for i, world in enumerate(validation_worlds):
                batch = torcher.targets_batch(problem, [world], dev, dtype=dtype)
                batch = batch.filter(lambda target: target[1] in chosen_targets)
                base_val = batch.base_val.to(dev)
                fuzzy_vals = dilp.infer(base_val, rulebook, weights = masked_softmax(fuzzy_p, rulebook.mask, softmax_temp), steps=val_steps, devices=devs, problem = problem, split=split)
                logging.info(f"{fuzzy_vals.mean()=}")

                crisp_vals = dilp.infer(base_val, rulebook, weights = crisp, steps=val_steps, devices=devs, problem = problem, split=split)

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

def adjust_weights(weights : List[torch.nn.Parameter]):
    with torch.no_grad():
            for w in weights:
                a = torch.max(torch.zeros(size=(), device=w.device), w)
                w[:] = a / a.sum(dim=1, keepdim=True)
                #assert (w.sum(-1) == 1).all(), f"{w.sum(-1)=}"

def norm_loss(weights : torch.Tensor) -> torch.Tensor:
    x = weights
    x = (x.softmax(-1) * x.log_softmax(-1))

    return -x.sum()


if __name__ == "__main__":
    fire.Fire(main)
