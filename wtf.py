import torch
import fire, tqdm #type: ignore

def main(epochs : int):
    p = torch.nn.Parameter(torch.rand(10))
    opt = torch.optim.SGD([p], lr=1e-2)
    for _ in tqdm.tqdm(range(0, epochs)):
        opt.zero_grad()
        x = p.softmax(-1)
        loss = -(p.softmax(-1) * p.log_softmax(-1)).sum()
        loss.backward()
        print(f"{p=} {p.grad=} {loss=}")
        opt.step()

if __name__ == "__main__":
    fire.Fire(main)