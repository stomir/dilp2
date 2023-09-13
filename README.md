# δILP2

This repository contains an implementation of the method outlined in "Differentiable Inductive Logic Programming in High-Dimensional Space" (see https://arxiv.org/abs/2208.06652). Our implementation is based on https://github.com/ai-systems/DILP-Core and the method presented in "Learning Explanatory Rules from Noisy Data" (see https://arxiv.org/abs/1711.04574). Our System extends δILP by large-scale predicate invention and removes almost all language bias.

## Usage

Example usage (with CUDA acceleration):

```bash
python3 run.py examples/arith_even --inv=20 --steps=25 --epochs=2000 --seed=1 --cuda=True
```

Using multiple GPUs:

```bash
python3 run.py examples/arith_even --inv=20 --steps=25 --epochs=2000 --seed=1 --devices=0,1,2,3
```

## Arguments

More flags can be found in the `run.py` file, as arguments of the `main` function:

```python
task : str, 
        epochs : int, steps : int, 
        batch_size : float = 0.5,
        cuda : Union[int,bool] = False,
        inv : int = 0,
        debug : bool = False,
        norm : str = 'mixed',
        optim : str = 'adam', lr : float = 0.05,
        clip : Optional[float] = None,
        info : bool = False,
        init : str = 'uniform',
        init_size : float = 10.0,        
        end_early : Optional[float] = 1e-3,
        seed : Optional[int] = None,
        validate : bool = True,
        validation_steps : Union[float,int] = 2.0,
        validate_training : bool = True,
        worlds_batch_size : int = 1,
        devices : Optional[List[int]] = None,
        input : Optional[str] = None, output : Optional[str] = None,
        checkpoint : Optional[str] = None,
        validate_on_cpu : bool = True,
        training_worlds : Optional[int] = None,
        softmax_temp : Optional[float] = 1.0,
        norm_p : float = 1.0,
        target_copies : int = 0,
        split : int = 2,
        min_parameter : Optional[float] = None,
        max_parameter : Optional[float] = None,
        compile : bool = False,
```
In particular, the argument split controls how weights are assigned, i.e. 0--per template, 1--per clause, and 2--per literal. Per template weight assignment was used by δILP and per literal is used by δILP2 by default. 
