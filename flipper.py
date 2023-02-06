import torch
from typing import *
import logging

class Flipper(torch.optim.Optimizer):
    def __init__(self, params : Iterable[torch.nn.Parameter], lr : float = 0, temp : Optional[float] = None):
        self.params : List[torch.nn.Parameter] = list(params)
        self.lr = lr
        self.temp = temp
        defaults : Dict[str,object] = dict(lr=lr, temp=temp)
        super(Flipper, self).__init__(params, defaults)
        
    def step(self, closure : Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        with torch.no_grad():
            for param in self.params:
                if param.grad is None:
                    continue
                
                x : torch.Tensor = param.grad
                logging.info(f'{param.grad.min()=}')
                x = torch.where(condition = (param == 0), input = -x, other = torch.as_tensor(-1000, device=param.device))
                max_x = x.max(dim = -1).values
                change_count = 1 if self.lr == 0 else int(max_x.numel() * self.lr)
                indices : torch.LongTensor = max_x.view(-1).sort(descending = True).indices
                changed_idxs = indices[:change_count]
                logging.debug(f"{indices=} {changed_idxs=}")
                if self.temp is None:
                    for bad_idx in changed_idxs.cpu():
                        n = bad_idx.item()
                        idx = (n >> 2, (n & 2) >> 1, n & 1)
                        val = torch.max(x[idx], dim=-1).indices
                        logging.info(f'changing {idx=} to {val} ({param[idx].max(-1).indices=})')
                        param[idx] = 0
                        param[idx][torch.max(x[idx], dim=-1).indices] = 1
                else:
                    raise NotImplementedError()
                
        return loss
        
        