# δILP2

a differentiable ILP system using high-deimensional search space

## Usage

Example usage (with CUDA acceleration):

```bash
python3 run.py examples/arith_even --inv 20 --steps 20 --epochs 1000 --batch_size 0.5 --cuda True
```

Using multiple GPUs:

```bash
python3 run.py examples/arith_even --inv 20 --steps 20 --epochs 1000 --batch_size 0.5 --cuda False --devices 0,1,2,3
```

## Arguments

More flags can be found in the `run.py` file, as arguments of the `main` function:

```python
        task : str, 
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
        checkpoint : Optional[str] = None,
        validate_on_cpu : bool = True,
        training_worlds : Optional[int] = None,
```

and in `torcher.py` as arguments of the `rules` function:

```python
            layers : Optional[List[int]] = None,
            unary : List[str] = [],
            recursion : bool = True, 
            invented_recursion : bool = True,
            full_rules : bool = False,
```

## Running in bulk

You can use `batchrun.sh` script to run the script multiple times using `slurm` and aggregate the results:

```bash
bash batchsize.sh 100 examples/arith_even --inv 20 --steps 20 --epochs 1000 --batch_size 0.5 --outdir even_results
```