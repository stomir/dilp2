import fire #type: ignore
from dilp import *
import torch
from tqdm import tqdm #type: ignore

def main(epochs : int = 100, steps : int = 5, cuda : bool = False):
    dev = torch.device(0) if cuda else torch.device('cpu')

    #logging.getLogger().setLevel(logging.DEBUG)
    base_val = torch.as_tensor([
        [ #false
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ], [ #succ
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1],
            [0,0,0,0,0]
        ], [ #p
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ], [ #q
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]
    ], dtype=torch.float, device=dev)
    
    rulebook = Rulebook(
        body_predicates=torch.as_tensor([
            [[[0,0] for _ in range(0,11)] for _ in range(0,2)],
            [[[0,0] for _ in range(0,11)] for _ in range(0,2)],
            [[[1,1],[1,1],[1,1],[1,1],[3,3],[2,0],[1,1],[0,0],[1,2],[2,1],[2,2]] for _ in range(0,2)],
            [[[1,1],[1,1],[1,1],[1,1],[3,3],[2,0],[1,1],[0,0],[1,2],[2,1],[2,2]] for _ in range(0,2)],
        ], device=dev),
        variable_choices=torch.as_tensor([
            [[[0,0] for _ in range(0,11)] for _ in range(0,2)],
            [[[0,0] for _ in range(0,11)] for _ in range(0,2)],
            [[[0,0],[1,0],[2,7],[5,4],[3,3],[2,0],[1,1],[0,0],[1,2],[2,1],[2,2]] for _ in range(0,2)],
            [[[0,0],[1,0],[2,7],[5,4],[3,3],[2,0],[1,1],[0,0],[1,2],[2,1],[2,2]] for _ in range(0,2)],
        ], device=dev)
    )
    logging.debug(f"{rulebook.body_predicates.shape=},{rulebook.variable_choices.shape=}")
    pred_names = ['false', 'succ', 'p', 'q']

    targets = torch.as_tensor([
        [2,0,2],
        [2,1,3],
        [2,2,4],
        [2,1,1],
        [2,3,2],
        [2,0,0],
        [2,0,1]
    ], device=dev)
    target_values = torch.as_tensor([
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0
    ], device=dev)

    #weights : torch.nn.Parameter = torch.nn.Parameter(torch.rand(size=(4,2,11), device=dev))
    weights : torch.nn.Parameter = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(4,2,11), device=dev))
    #weights = torch.nn.Parameter(torch.zeros(4,2,11))
    #with torch.no_grad():
    #    weights[2][0][2] = 1000
    #    weights[2][1][0] = 1000

    opt = torch.optim.RMSprop([weights], lr=1e-2)
    #opt = torch.optim.SGD([weights], lr=1e-2)
    for epoch in tqdm(range(0, epochs)):
        opt.zero_grad()
        mse_loss = loss(base_val, rulebook=rulebook, weights = weights, targets=targets, target_values=target_values, steps=steps)
        mse_loss.backward()
        with torch.no_grad():
            print(f"{weights.grad[2]}")
            #weights.grad = weights.grad / torch.max(weights.grad.norm(2, dim=-1, keepdim=True), torch.as_tensor(1e-8))
            pass
        opt.step()
        print(f"loss: {mse_loss.item()}")
        print(f"{weights[2]=}")

    print_program(rulebook, weights, pred_names)

if __name__ == "__main__":
    fire.Fire(main)
