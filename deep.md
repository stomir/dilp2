srun --pty -p IFItitan -c 31 python3 sample.py examples/member --inv 30 --epochs 500 --cuda True --norm mixed --optim adam --steps 15 --seed None --normalize_threshold=1e-2 --batch_size=5 --seed 1 --layers 30,

srun --pty -p IFItitan -c 31 python3 sample.py examples/member --inv 30 --epochs 500 --cuda True --norm mixed --optim adam --steps 15 --seed None --normalize_threshold=1e-2 --batch_size=5 --seed 1
