# import packages
import numpy as np
from pypher.pypher import psf2otf
from tqdm import tqdm
import torch
from .priors import priors

def deconv_adam_tv(b, c, lam, num_iters, learning_rate=5e-2, prior_name="anisotropic_tv",device=torch.device('cuda:0'),**kwargs):
    # check if GPU is available, otherwise use CPU

    # otf of blur kernel and forward image formation model
    cFT = psf2otf(c, np.shape(b))
    cFT = torch.from_numpy(cFT).to(device)
    Afun = lambda x: torch.real(torch.fft.ifft2(torch.fft.fft2(x) * cFT))


    # convert b to PyTorch tensor
    b = torch.from_numpy(b).to(device)
    h,w = b.shape
    # initialize x and convert to PyTorch tensor
    x = torch.zeros_like(b, requires_grad=True).to(device)
    # x = torch.distributions.uniform.Uniform(0,1).sample([h,w]).to(device)
    x.requires_grad = True 

    # initialize Adam optimizer
    optim = torch.optim.Adam(params=[x], lr=learning_rate)

    for it in range(num_iters):

        # set all gradients of the computational graph to 0
        optim.zero_grad()

        # this term computes the data fidelity term of the loss function
        loss_data = (Afun(x) - b).pow(2).sum()

        loss_regularizer = priors[prior_name](x=x,b=b,**kwargs)

        # compute weighted sum of data fidelity and regularization term
        loss = loss_data + lam * loss_regularizer

        # compute backwards pass
        loss.backward()

        # take a step with the Adam optimizer
        optim.step()

    # return the result as a numpy array
    return x.detach().cpu().numpy()
