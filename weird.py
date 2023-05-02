import torch

import torch
import math
from typing import *

class WeirdMin1(torch.autograd.Function):
    """
    max, but gradient is always progragated through the first argument
    """

    @staticmethod
    def forward(ctx, a,b):

        return torch.min(a,b)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, torch.zeros_like(grad_output)

weirdMin1 : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] 
weirdMin1 = WeirdMin1.apply #type: ignore

class WeirdMax(torch.autograd.Function):
    """
    max, but gradient is always progragated through both arguments
    """

    @staticmethod
    def forward(ctx, a,b):

        return torch.max(a,b)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output / 2, grad_output / 2

class WeirdMax2(torch.autograd.Function):
    """
    max, but gradient is proportional to input
    """
    @staticmethod
    def forward(ctx, a,b):
        s = a + b
        s = torch.max(s,  torch.as_tensor(1e-9, device=a.device))
        ctx.sa = a / s
        ctx.sb = b / s
        return torch.max(a,b)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.sa, grad_output * ctx.sb

class WeirdMin(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a,b):

        return torch.min(a,b)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, grad_output

class WeirdMinDim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, dim):
        ctx.dim = dim
        ctx.size = a.shape[dim]
        return  a.min(dim=dim).values

    @staticmethod
    def backward(ctx, grad_output):
        size = len(grad_output.shape)+1
        rep = [ctx.size if d in {ctx.dim, ctx.dim + size} else 1 for d in range(0, len(grad_output.shape)+1)]
        return grad_output.unsqueeze(ctx.dim).repeat(rep), None

class WeirdMaxDim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, dim):
        ctx.dim = dim
        ctx.size = a.shape[dim]
        return  a.max(dim=dim).values

    @staticmethod
    def backward(ctx, grad_output):
        size = len(grad_output.shape)+1
        rep = [ctx.size if d in {ctx.dim, ctx.dim + size} else 1 for d in range(0, len(grad_output.shape)+1)]
        return grad_output.unsqueeze(ctx.dim).repeat(rep), None

class WeirdMax2Dim(torch.autograd.Function):
    eps = 1e-9

    @staticmethod
    def forward(ctx, a, dim):
        ctx.dim = dim
        ctx.scale = a / torch.max(a.sum(dim=dim,keepdim=True), torch.as_tensor(1e-9, device=a.device))
        return  a.max(dim).values

    @staticmethod
    def backward(ctx, grad_output):
        x = (grad_output.unsqueeze(ctx.dim) * ctx.scale)
        return x, None

class WeirdMax2BDim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, dim):
        ctx.dim = dim
        am = a.max(dim=dim,keepdim=True).values
        #x = a / am
        x = a
        x = x / x.sum(dim=dim,keepdim=True)
        ctx.scale = x
        return  am.squeeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        x = (grad_output.unsqueeze(ctx.dim) * ctx.scale)
        return x, None
    
class WeirdNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, dims, scale):
        ctx.dims = dims #sorted(dims, reverse=True)
        ctx.scale = scale
        return a

    @staticmethod
    def backward(ctx, grad_output):
        mean = grad_output.mean(ctx.dims, keepdim=True)
        var = (grad_output - mean).square().mean().sqrt()
        eps = 1e-9
        return grad_output / (ctx.scale * (var + eps)), None, None

class GaussMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, sigma):
        ctx.x = phi(a-b, sigma=sigma)
        return torch.max(a,b)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.x, grad_output * (1-ctx.x), None

def phi(x, sigma=1.0):
    return (1+torch.special.erf(x * 1/math.sqrt(2)/sigma))/2

def gauss_max(sigma=2.0) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return lambda a, b: GaussMax.apply(a, b, sigma)