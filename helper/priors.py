###################################################
# All different kind of priors. Refer to the report for their formula
###################################################

# import packages
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent/"training"))

import cv2
import numpy as np
from pypher.pypher import psf2otf
from tqdm import tqdm
import torch
import torchvision
from torch.autograd import Variable

import trainer
import utils
from helper.CannyEdgePytorch.net_canny import Net

# transform = torchvision.transforms.ToTensor()

def canny(x):
    if x.device.type=="cpu":
        use_cuda = False
    else:
        use_cuda=True
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.to(x.device)
    net.eval()
    batch = torch.stack([x]).float()
    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).to(x.device)

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)
    return (thresholded[0,0]>7)*1.0


def grad_image(x):
    # finite differences kernels and corresponding otfs
    dx = np.array([[-1., 1.]])
    dy = np.array([[-1.], [1.]])
    dxFT = torch.from_numpy(psf2otf(dx, x.shape)).to(x.device)
    dyFT = torch.from_numpy(psf2otf(dy, x.shape)).to(x.device)
    dxyFT = torch.stack((dxFT, dyFT), axis=0)
    return torch.stack((torch.real(torch.fft.ifft2(torch.fft.fft2(x) * dxFT)),torch.real(torch.fft.ifft2(torch.fft.fft2(x) * dyFT))),axis=0)

def hessian(x):
    grads = grad_image(x)
    dxx,dxy = grad_image(grads[0])
    dyx,dyy = grad_image(grads[1])
    return torch.stack((dxx,dxy,dyx,dyy),axis=0)

def anisotropic_tv(x,**kwargs):
     grads = grad_image(x)
     return torch.sum(torch.abs(grads[0])) + torch.sum(torch.abs(grads[1])) 

def isotropic_tv(x,**kwargs):
    grads = grad_image(x)
    return torch.sum(torch.norm(grads,dim=0))

def l1norm(x,**kwargs):
    return torch.norm(x,p=1)

def l2norm(x,**kwargs):
    return torch.norm(x)

def laplacian(x,**kwargs):
    hess = hessian(x)
    lap = hess[0] + hess[-1]
    return torch.norm(lap)**2

def hessian_schatten_norm(x,**kwargs):
    #We only consider the 2nd order schatten norm which is frobenius norm of the hessian
    hess = hessian(x)
    return torch.sum(torch.norm(hess,dim=0))

def maximize_cells(x,**kwargs):
    x_mod = torch.stack((x,x,x))
    edge = canny(x_mod)
    h,w = edge.shape
    #The cells are approxiamately of diameter 8micrometer, which is 16 pixels for images at 0.5 micrometer/pixel resolution
    radius = 8
    filter_radius = radius + 2
    filter_size = filter_radius * 2 + 1
    
    #Circle filter
    img_filter = np.zeros((3, filter_size, filter_size))
    cv2.circle(img_filter[0], (filter_radius, filter_radius), int(radius / 2), -1, -1)
    cv2.circle(img_filter[0], (filter_radius, filter_radius), radius, 1, 3)

    #Ellipse filter
    e = 0.75
    cv2.ellipse(img_filter[1], (filter_radius, filter_radius), (int(filter_radius), int(np.sqrt(1-e**2)*filter_radius)), 45, 0, 360, 1, 2)
    cv2.ellipse(img_filter[1], (filter_radius, filter_radius), (int(filter_radius/2), int(np.sqrt(1-e**2)*filter_radius/2)), 45, 0, 360, -1, -1)
    
    cv2.ellipse(img_filter[2], (filter_radius, filter_radius), (int(filter_radius), int(np.sqrt(1-e**2)*filter_radius)), 360-45, 0, 360, 1, 2)
    cv2.ellipse(img_filter[2], (filter_radius, filter_radius), (int(filter_radius/2), int(np.sqrt(1-e**2)*filter_radius/2)), 360-45, 0, 360, -1, -1)

    center_cal = 0
    for i in range(3):
        filter_custom = torch.from_numpy(psf2otf(img_filter[i], edge.shape)).to(edge.device)
        filter_result = torch.real(torch.fft.ifft2(torch.fft.fft2(edge) * filter_custom))
        min_val, max_val = filter_result.min(), filter_result.max()
        centers = filter_result>(min_val+max_val)*0.7
        #convolution in fourier domain leads to undesired boundary effects
        centers[-8:-1,:] = 0
        centers[0:8,:] = 0
        centers[:,0:8] = 0
        centers[:,-8:-1] = 0
        center_cal+=(-torch.sum(centers)/h)
    return center_cal

def no_prior(x,**kwargs):
    return 0

def cross_entropy(x,**kwargs):
    # 1: without blur
    # 0: with blur
    py_x = torch.nn.functional.softmax(kwargs["model"](torch.unsqueeze(torch.stack((x.float(),x.float(),x.float())),0)))[0][1]
    return -torch.log(py_x)

def kl_divergence(x,b,**kwargs):
    p = torch.nn.functional.softmax(kwargs["model"](torch.unsqueeze(torch.stack((x.float(),x.float(),x.float())),0)))[0]
    kl_div = p[1]-torch.log(torch.exp(p[0]))
    return -kl_div

priors = {"anisotropic_tv": anisotropic_tv,
          "isotropic_tv": isotropic_tv,
          "l1": l1norm,
          "l2": l2norm,
          "laplacian": laplacian,
          "hessian_schatten": hessian_schatten_norm,
          "maximize_cells":maximize_cells,
          "no_prior":no_prior,
          "cross_entropy":cross_entropy,
          "kl_divergence":kl_divergence}