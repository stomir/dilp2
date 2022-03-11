# Hypotheses list

## Does the main approach work?

- does it generalize well

## Is too much prediates harmful?

fizz, 20 steps:

| Invented | Valid seeds | Steps |
|----------|-------------|-------|
| 10       | `07/20`     | 20    |
| 20       | `14/20`     | 20    |
| 40       | `15/20`     | 20    |
| 50       | `14/20`     | 20    |
| 50       | `17/20`     | 30    |
| 70       | `15/20`     | 20    |

## Is batching helpful?

YES, at least when examples are unbalanced: `14/20` vs `9/20`

```bash
bash batchrun.sh examples/fizz/ 20 --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 > fizz_30_20_a_batched
bash batchrun.sh examples/fizz/ 20 --inv 30 --steps 20 --epochs 1000 --cuda True > fizz_30_20_a_unbatched
```

## Is deep better than flat?

YES: `14/20` vs `4/20`

```bash
bash batchrun.sh examples/fizz/ 20 --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 > fizz_30_20_a_batched
bash batchrun.sh examples/fizz/ 20 --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --layers 15,15 > fizz_30_20_b_flat2
```

## Gradient normalization beneficial?

- 1e2 seems to be good for `even`
  
## Is entropy loss beneficial?

| Description | Run | Result |
|-------------|-----|--------|
| entropy loss `1` from the start | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --batch_size 3 --normalize_threshold None` | `00/20` |
| entropy loss `1e-2` from the start | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --normalize_threshold None --norm_weight 1e-2` | `00/20` |
| entropy loss `1e-4` from the start | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --normalize_threshold None --norm_weight 1e-4` | `13/20` |
| entropy loss `1e-6` from the start | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --normalize_threshold None --norm_weight 1e-6` | `12/20` |
| no entropy loss | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --norm_weight 0` | `12/20` |
| entropy enabled at `1e-2` target loss | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --batch_size 3` | `12/20` |
| entropy `1e-4` enabled at `0.1` target loss | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-4` | `13/20` |
| entropy `1e-4` enabled at `0.1` target loss with gradient norm `1e2` | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-4 --normalize_gradients 1e2` | `14/20` |
| entropy `1e-4` enabled at `0.1` target loss with gradient norm `1e2` | `examples/fizz/ --inv 30 --steps 20 --epochs 1000 --cuda True --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-4 --normalize_gradients 1e2` | `14/20` |
| longer entropy `1e-4` enabled at `0.1` target loss with gradient norm `1e2` | `examples/fizz/ --inv 30 --steps 20 --epochs 2000 --cuda True --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-4 --normalize_gradients 1e2 --end_early 1e-4` | `13/19` |
| no entropy, longer | `examples/fizz/ --inv 30 --steps 20 --epochs 2000 --cuda True --batch_size 3 --norm_weight 0 --end_early 1e-4` | `18/20` |
| no entropy, even longer | `examples/fizz/ --inv 30 --steps 20 --epochs 4000 --cuda True --batch_size 3 --norm_weight 0 --end_early 1e-4` | `13/20` |
| no entropy, longer, repeated | `examples/fizz/ --inv 30 --steps 20 --epochs 2000 --cuda True --batch_size 3 --norm_weight 0 --end_early 1e-4` | `12/20` |
| | `1 20 examples/fizz/ --inv 30 --steps 20 --epochs 4000 --cuda True --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-4 --normalize_gradients 1e2 --end_early 1e-4` | `16/20` |
| long no entropy | `1 20 examples/fizz/ --inv 30 --steps 20 --epochs 4000 --cuda True --batch_size 3 --norm_weight 0 --end_early 1e-4` | `13/20` |
| | `110 119 examples/fizz/ --inv 30 --steps 20 --epochs 4000 --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-2 --normalize_gradients 1e2 --end_early 1e-4` | `17/20` |


| Description | Run | Result | Fuzzy result |
|-------------|-----|--------|--------------|
| no entropy | `examples/fizz/ --inv 30 --steps 20 --epochs 4000 --batch_size 3 --normalize_threshold None --norm_weight 0 --end_early 1e-4` | `14/20` | `18/20` |
| entropy `1e-2` enabled at target loss `0.1`, gradients normalized to `100` | `examples/fizz/ --inv 30 --steps 20 --epochs 4000 --batch_size 3 --normalize_threshold 1e-1 --norm_weight 1e-2 --normalize_gradients 1e2 --end_early 1e-4` | `13/20` | `17/20` |



inverted at the start?

## Entropy loss where there is gradient?

## Distance loss from bad spot?

## Different norms?

parametrized Łukasiewicz norms from some paper?

## Some loss to prefer one body predicate (for grandparent)?
