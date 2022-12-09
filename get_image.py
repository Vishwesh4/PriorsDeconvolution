# import packages
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent/"training"))
import random

import argparse
import numpy as np
from numpy.fft import fft2, ifft2
import pandas as pd
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.filters import gaussian
from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

# import our Adam-based deconvolution code
from helper.deconv_adam import *
import trainer
import training.utils

# random.seed(2022)

# helper function for computing a 2D Gaussian convolution kernel
def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)

def deconvolve(img,b,c,lam,num_iters,learning_rate,prior_name,model):
    x_adam_prior = np.zeros(np.shape(b))
    for it in range(3):
        x_adam_prior[:, :, it] = deconv_adam_tv(b[:, :, it], c, lam, num_iters, learning_rate, prior_name, device, model=model)
    # clip results to make sure it's within the range [0,1]
    x_adam_prior = np.clip(x_adam_prior, 0.0, 1.0)
    # compute PSNR using skimage library and round it to 2 digits
    PSNR_ADAM_PRIOR = round(peak_signal_noise_ratio(img, x_adam_prior), 1)
    SSIM_ADAM_PRIOR = structural_similarity(img,x_adam_prior,data_range=x_adam_prior.max()-x_adam_prior.min(),multichannel=True)
    MSE_ADAM_PRIOR = mean_squared_error(img,x_adam_prior)
    return x_adam_prior, PSNR_ADAM_PRIOR, SSIM_ADAM_PRIOR, MSE_ADAM_PRIOR

def blur_image(img):
    # simulated measurements for all 3 color channels
    b = np.zeros(np.shape(img))
    for it in range(3):
        b[:, :, it] = blur_kernel(img[:, :, it]) + NOISE_SIGMA * np.random.randn(img.shape[0], img.shape[1])
    return b

def calc_metrics_all_priors(img, b, c, prior_list, lam, num_iters, learning_rate,model):
    """
    Deconvolves and calculates metrics for all priors
    img: true image
    b: blurred image
    c: blur kernel
    lam: weighting of prior term
    model: model for cross entropy and kl divergence prior
    """
    x_all = []
    PSNRS = []
    MSES = []
    SSIMS = []
    for i in range(len(prior_list)):
        # run PyTorch-based Adam solver for each color channel with different regularizers
        img_run = img.copy()
        b_run = b.copy()
        x_adam, PSNR_adam, SSIM_adam, MSE_adam = deconvolve(img_run,b_run,c,lam,num_iters,learning_rate,prior_list[i],model=model)
        x_all.append(x_adam)
        PSNRS.append(PSNR_adam)
        SSIMS.append(SSIM_adam)
        MSES.append(MSE_adam)
    return x_all, PSNRS, SSIMS, MSES



if __name__=="__main__":
    #For different experiments
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",help="blur kernel set select, 0,1,or 2", type=int, required=True)
    parser.add_argument("-l",help="Lamda set select, 0,1", type=int, required=True)
    parser.add_argument("-n",help="Noise level set select, 0,1", type=int, required=True)

    args = parser.parse_args()
    
    gpu_id = 0
    
    #Hyperparameters
    BLUR_KERNEL_SET_ALL = [(10,1.5),(30,4.5),(60,6.5)]
    LAM_ALL = [0.05,0.5]
    NOISE_ALL = [0.15,0.05]
    
    idx_blur_kernel = args.b
    idx_lam = args.l
    idx_noise = args.n
    
    BLUR_KERNEL_SET = BLUR_KERNEL_SET_ALL[idx_blur_kernel]
    BLUR_SIZE = (BLUR_KERNEL_SET[0],BLUR_KERNEL_SET[0])
    BLUR_SIGMA = BLUR_KERNEL_SET[1]
    NOISE_SIGMA = NOISE_ALL[idx_noise]
    # NOISE_SIGMA = 0.05
    
    SAMPLE = 150
    LAM = LAM_ALL[idx_lam]  # relative weight of prior term
    NUM_ITERS = 75          # number of iterations for Adam
    LEARNING_RATE = 5e-2    # learning rate
    FILE_NAME = f"./results/fullexp_{BLUR_SIZE[0]}_{BLUR_SIGMA}_{NOISE_SIGMA}_{LAM}.npy"
    PRIOR_LIST = ["no_prior","anisotropic_tv","isotropic_tv","hessian_schatten","l1","l2","laplacian","maximize_cells","cross_entropy","kl_divergence"]


    print("Selected Hyperparameters:")
    print(f"Blur Kernel: {BLUR_KERNEL_SET}\nNoise Sigma: {NOISE_SIGMA}\nLambda: {LAM}")
    print(f"Priors order: {PRIOR_LIST}")
    
    #Different models for different blur and sigma values
    #10, 1.5, 0.15
    if idx_blur_kernel==0:
        MODEL_PATH = "./training/Results/ce_train_v4_gray_10/saved_models/Checkpoint_06Dec12_24_30_1.00.pt"
    #30, 4.5, 0.15
    elif idx_blur_kernel==1:
        MODEL_PATH = "./training/Results/ce_train_v4_gray_30/saved_models/Checkpoint_06Dec12_34_25_1.00.pt"
    #60, 6.5, 0.15
    elif idx_blur_kernel==2:
        MODEL_PATH = "./training/Results/ce_train_v4_gray_60/saved_models/Checkpoint_06Dec12_32_15_1.00.pt"
    else:
        raise ValueError

    print(f"Model path: {MODEL_PATH}")

    #Load test dataset, we dont use the blur size, sigma, and blur sigma
    DATASET={
    "subclass_name": "ce_prior",
    "path": "./dataset",
    "mask_pth": "./dataset/masks",
    "blur_size": BLUR_SIZE[0],
    "blur_sigma": BLUR_SIGMA,
    "sigma": NOISE_SIGMA,
    "train_batch_size": 32,
    "test_batch_size": 32,
    "tile_h": 256,
    "tile_w": 256,
    "tile_stride_factor_w": 5,
    "tile_stride_factor_h": 5,
    "lwst_level_idx": 0,
    "threshold": 0.7
    }

    #Load model for CE/KL priors
    device = torch.device(f"cuda:{gpu_id}")
    model = trainer.Model.create(subclass_name="ce_prior")
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    #disable gradient descent for the model
    for params in model.parameters():
        params.requires_grad=False

    #Load dataset
    fulldataset = trainer.Dataset.create(**DATASET)
    dataset = fulldataset.testset.image_dataset

    c = fspecial_gaussian_2d(BLUR_SIZE, BLUR_SIGMA)
    cFT = psf2otf(c, (dataset[0].shape[0], dataset[0].shape[1]))
    blur_kernel = lambda x: np.real(ifft2(fft2(x) * cFT))


    #Run experiment over an image
    img_idx = 56
    img = dataset[img_idx]/255
    b = blur_image(img)
    x_all,psnrs,ssims,mses = calc_metrics_all_priors(img, b, c, PRIOR_LIST, LAM, NUM_ITERS, LEARNING_RATE, model)

    img_dir_path = f"./results/priors_{img_idx}"
    os.mkdir(img_dir_path)
    fig = plt.figure()
    plt.imshow(img)
    plt.title("Target Image", fontsize=10)
    plt.axis("off")
    plt.savefig(Path(img_dir_path)/"image.png")

    fig = plt.figure()
    plt.imshow(b)
    plt.title("Blurry and Noisy Image,\nPSNR: {:.2f}\nSSIM: {:.2f}".format(peak_signal_noise_ratio(img, b),structural_similarity(img,b,data_range=b.max()-b.min(),multichannel=True)), fontsize=10)
    plt.axis("off")
    plt.savefig(Path(img_dir_path)/"blurimage.png")

    for i in range(len(PRIOR_LIST)):
        fig = plt.figure()
        plt.imshow(x_all[i])
        plt.title("{},\nPSNR: {:.2f}\nSSIM: {:.2f}".format(PRIOR_LIST[i],psnrs[i],ssims[i]), fontsize=10)
        plt.axis("off") 
        plt.savefig(Path(img_dir_path)/f"{PRIOR_LIST[i]}.png")
