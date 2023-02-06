import torch
import numpy
import matplotlib.pyplot as plt #type: ignore
import os
from typing import *

def line_plot(data : torch.Tensor, label : str = 'line plot', file : Optional[str] = None):
    plt.plot(data.detach().view(-1).cpu().numpy())
    plt.title(label)
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
    plt.clf()

def mkdir(*args):
    path = os.path.join(*args)
    if not os.path.isdir(path):
        os.mkdir(path)

def weights_plot(weights : torch.Tensor, outdir : str, epoch : int):
    mkdir(outdir)
    for p in range(weights.shape[0]):
        mkdir(outdir, f"p{p}")
        for clause in range(2):
            mkdir(outdir, f"p{p}", f"clause{clause}")
            for body_pred in range(2):
                mkdir(outdir, f"p{p}", f"clause{clause}", f"bp{body_pred}")
                line_plot(data = weights[p][clause][body_pred].detach().sort()[0],
                            file = os.path.join(outdir, f"p{p}", f"clause{clause}", f"bp{body_pred}", f"e{epoch:010}"),
                            label = f"epoch {epoch}")