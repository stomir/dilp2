import torch

import torch
import math
from typing import *

class WeirdMin1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a,b):

        return torch.min(a,b)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, torch.zeros_like(grad_output)

weirdMin1 : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] 
weirdMin1 = WeirdMin1.apply #type: ignore

class WeirdMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a,b):

        return torch.max(a,b)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, grad_output

class WeirdMax2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a,b):
        m = torch.max(a,b)
        sa = a / m
        sb = b / m
        s = sa + sb
        ctx.sa = sa / s
        ctx.sb = sb / s
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

    @staticmethod
    def forward(ctx, a, dim):
        ctx.dim = dim
        am = a.max(dim=dim,keepdim=True).values
        x = a / am
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