from sqlite3 import DataError
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
        batch_size : Optional[Union[int, float]],
        cuda : Union[int,bool] = False, inv : int = 0,
        debug : bool = False, norm : str = 'mixed',
        entropy_weight : float = 0.0,
        optim : str = 'adam', lr : float = 0.05, clip : Optional[float] = None,
        info : bool = False,
        entropy_enable_threshold : Optional[float] = 1e-2,
        normalize_gradients : Optional[float] = None,
        init : str = 'normal',
        init_size : float = 1,        
        entropy_weight_step = 1,
        end_early : Optional[float] = 1e-3,
        seed : Optional[int] = None,
        validate : bool = True,
        validation_steps : Optional[int] = None,
        validate_training : bool = True,
        worlds_batch_size : int = 1,
        devices : Optional[List[int]] = None,
        entropy_gradient_ratio : Optional[float] = None,
        input : Optional[str] = None, output : Optional[str] = None,
        tensorboard : Optional[str] = None,
        use_float64 : bool = False,
        checkpoint : Optional[str] = None,
        validate_on_cpu : bool = True,
        training_worlds : Optional[int] = None,
        **rules_args):
    if info:
        logging.getLogger().setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    inv = int(inv)
    steps = int(steps)


    if use_float64:
        torch.set_default_tensor_type(torch.DoubleTensor)

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

    dilp.set_norm(norm)
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

    train_worlds = [loader.load_world(os.path.join(task, d), problem = problem, train = True) for d in dirs if d.startswith('train')]
    validation_worlds = [loader.load_world(os.path.join(task, d), problem = problem, train = False) for d in dirs if d.startswith('val')] \
                        + (train_worlds if validate_training else [])
                        
    if training_worlds is not None:
        train_worlds = train_worlds[:training_worlds]

    worlds_batches : Sequence[torcher.WorldsBatch] = [torcher.targets_batch(problem, worlds, dev) for worlds in torcher.chunks(worlds_batch_size, train_worlds)]
    choices : Sequence[numpy.ndarray] = [numpy.concatenate([numpy.repeat(i, len(batch.targets(target_type).idxs)) for i, batch in enumerate(worlds_batches)]) for target_type in loader.TargetType]

    #This should not be used. Instead one of the dictionaries should be used.
    #pred_names = list(pred_dict_rev.keys())

    rulebook = torcher.rules(problem, dev, **rules_args)

    shape = rulebook.mask.shape

    assert init in {'normal', 'uniform'}
    weights : torch.nn.Parameter = torch.nn.Parameter(torch.normal(mean=torch.zeros(size=[shape[0], 2, 2, shape[3]], device=dev), std=init_size)) \
        if init == 'normal' else torch.nn.Parameter(torch.rand([shape[0], 2, 2, shape[3]], device=dev) * init_size)
    epoch : int = 0

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

    if input is not None:
        w, opt_sd, epoch, entropy_enabled, entropy_weight_in_use = torch.load(input)
        with torch.no_grad():
            weights[:] = w.to(dev)
        opt.load_state_dict(opt_sd)
        logging.info(f'loaded weights from {input}')
        del w, opt_sd
        

    for _ in (tq := tqdm(range(0, int(epochs)))):
        epoch += 1
        opt.zero_grad()

        if type(batch_size) is int:
            chosen_worlds_batches = [numpy.random.choice(choices_of_type, replace=False, size=batch_size) for choices_of_type in choices]
        elif type(batch_size) is float:
            chosen_per_world_batch : Dict[loader.TargetType, Sequence[torch.Tensor]] = dict((ttype, [(torch.rand(len(batch.targets(ttype)), device=dev) <= batch_size) for batch in worlds_batches]) for ttype in loader.TargetType)
            chosen_per_ttype : Dict[loader.TargetType, int] = dict((ttype, sum(int(c.sum().item()) for c in chosen_per_world_batch[ttype])) for ttype in loader.TargetType)


        all_worlds_sizes = dict((t, sum(len(b.targets(t)) for b in worlds_batches)) for t in loader.TargetType)

        loss_sum = 0.0

        try:
            target_losses : List[float] = []
            for i, batch in enumerate(worlds_batches):
                ws = masked_softmax(weights, rulebook.mask)

                assert (ws < 0).sum() == 0 and (ws > 1).sum() == 0, f"{ws=} {i=}"

                vals = dilp.infer(base_val = batch.base_val, rulebook = rulebook, 
                            weights = ws, steps=steps, devices = devs)

                assert (vals < 0).sum() == 0 and (vals > 1).sum() == 0, f"{(vals < 0).sum()=} {(vals > 1).sum()=} {i=} {steps=} {devices=} {vals.max()=}"

                if type(batch_size) is int:
                    assert False, "not currently supported"
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
                    one_target_loss = 0.0
                    for target_type in loader.TargetType:
                        targets = batch.targets(target_type).idxs.to(dev, non_blocking=False)
                        preds = dilp.extract_targets(vals, targets)
                        loss = dilp.loss(preds, target_type, reduce=False)

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

                logging.debug(f"{ls=} {ls.item()=}")
                if ls.item() != 0:
                    ls.backward()
                assert ls >= 0
                loss_sum += ls.item()

                del loss, vals, targets, preds, ls
                torch.cuda.empty_cache()
            
            # if normalize_gradients is not None:
            #     with torch.no_grad():
            #         for fuzzy in [weights]:
            #             #w /= w.sum(-1, keepdim=True) * w.sign()
            #             if fuzzy.grad is not None:
            #                 fuzzy.grad[:] = torch.nn.functional.normalize(fuzzy.grad, dim=-1)
            #                 fuzzy.grad *= normalize_gradients
                
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

            target_loss = sum(target_losses) / len(target_losses)
            if end_early is not None and target_loss < end_early:
                break

            opt.step()
            #adjust_weights(weights)
        except AssertionError as e:
            weights_file : str = output + "_faulty" if output is not None else "faulty_weights"
            logging.error(f"assertion during backprop, saving weights to {weights_file}\n{traceback.format_exc()}")
            torch.save(weights.detach(), weights_file)
            raise e
        
        tq.set_postfix(entropy = actual_entropy.item(), batch_loss = loss_sum, entropy_weight=entropy_weight_in_use * entropy_weight, target_loss = target_loss)
        if tb is not None:
            tb.add_scalars("train", 
                {'entropy' : actual_entropy.item(), 'batch_loss' : loss_sum, 
                'target_loss' : target_loss,
                'entropy_weight' : entropy_weight_in_use * entropy_weight},
                global_step=epoch)

    dilp.print_program(problem, mask(weights, rulebook))
    
    if validate:
        last_target = target_loss
        last_entropy = actual_entropy.item()
        with torch.no_grad():
            total_loss = 0.0
            total_fuzzy = 0.0
            valid_worlds = 0
            fuzzily_valid_worlds = 0
            valid_tr_worlds = 0
            fuzzily_valid_tr_worlds = 0
            if validate_on_cpu:
                dev = torch.device('cpu')
                devs = None
            rulebook = rulebook.to(dev, non_blocking=False)
            fuzzy = weights.detach().to(dev, non_blocking=False)
            crisp = mask(torch.nn.functional.one_hot(fuzzy.max(-1)[1], fuzzy.shape[-1]).to(dev).float(), rulebook)
            if validation_steps is None:
                validation_steps = steps * 2
            for i, world in enumerate(validation_worlds):
                base_val = torcher.base_val(problem, [world]).to(dev)
                batch = torcher.targets_batch(problem, [world], dev)
                fuzzy_vals = dilp.infer(base_val, rulebook, weights = masked_softmax(fuzzy, rulebook.mask), steps=validation_steps, devices=devs)
                fuzzy_loss : torch.Tensor = sum((dilp.loss(dilp.extract_targets(fuzzy_vals, batch.targets(target_type).idxs), target_type) for target_type in loader.TargetType), start=torch.as_tensor(0.0))

                crisp_vals = dilp.infer(base_val, rulebook, weights = crisp, steps=validation_steps, devices=devs)
                crisp_loss : torch.Tensor = sum((dilp.loss(dilp.extract_targets(crisp_vals, batch.targets(target_type).idxs), target_type) for target_type in loader.TargetType), start=torch.as_tensor(0.0))

                report = report_tensor([fuzzy_vals, crisp_vals], batch)
                fuzzy_report = report[:,1]
                crisp_report = report[:,2]
                target_values = report[:,0]
                fuzzy_acc = (fuzzy_report.round() == target_values).float().mean().item()
                crisp_acc = (crisp_report == target_values).float().mean().item()
                logging.info(f'world {i} {world.dir=} {fuzzy_acc=} {crisp_acc=}\n{report.numpy()}')
                total_loss += crisp_loss.item()
                total_fuzzy += fuzzy_loss.item()
                if crisp_acc == 1.0:
                    valid_worlds += 1
                    if world.train:
                        valid_tr_worlds += 1
                if fuzzy_acc == 1.0:
                    fuzzily_valid_worlds += 1
                    if world.train:
                        fuzzily_valid_tr_worlds += 1
            
            #valid_worlds /= len(validation_worlds)
            #fuzzily_valid_worlds /= len(validation_worlds)
            result ='     OK      ' if valid_worlds == len(validation_worlds) else \
                    '    FUZZY    ' if fuzzily_valid_worlds == len(validation_worlds) else \
                    '   OVERFIT   ' if valid_tr_worlds == len(train_worlds) else \
                    'FUZZY OVERFIT' if fuzzily_valid_tr_worlds == len(train_worlds) else \
                    '    FAIL     '
            print(f'result: {result} {valid_worlds=} {fuzzily_valid_worlds=} {total_loss=} ' +
                      f'{total_fuzzy=} {last_target=} {last_entropy=} {epoch=}')
    
    if output is not None:
        torch.save((weights.detach().cpu(), opt.state_dict(), epoch, entropy_enabled, entropy_weight_in_use), output)
        logging.info(f"saved weights to {output}")

    #END

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
