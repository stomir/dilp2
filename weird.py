import torch

import torch
import math


class WeirdMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a,b):

        return torch.max(a,b)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, grad_output

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