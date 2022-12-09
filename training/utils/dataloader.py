from pathlib import Path
import sys
import random
sys.path.append(str(Path(__file__).resolve().parent.parent.parent/"helper"))

import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from pypher.pypher import psf2otf
from PIL import Image
import cv2
import random
from numpy.fft import fft2, ifft2

from extract_patches import ExtractPatches

class BlurLoader(ExtractPatches):
    """ For loading patches across the whole slide and blur the image randomly"""
    def __init__(self,
                blur_kernel,
                sigma,
                image_pth,
                tile_h,
                tile_w,
                tile_stride_factor_h,
                tile_stride_factor_w,
                mode="train",
                mask_pth=None,
                output_pth=None,
                lwst_level_idx=0,
                threshold=0.7,
                transform=None):
        """
        Args:
            blur_kernel: The kernel to blur image with
            image_pth (str): path to wsi/folder of wsi.
            mask_pth(str): path to mask folder
            tile_h (int): tile height
            tile_w (int): tile width
            tile_stride_factor_h (int): stride height factor, height will be tile_height * factor
            tile_stride_factor_w (int): stride width factor, width will be tile_width * factor
            lwst_level_idx (int): lowest level for patch indexing
            mode (str): train or val, split the slides into trainset and val set
        """
        self.mode = mode
        self.transforms = transform
        self.sigma = sigma

        super().__init__(image_pth, tile_h, tile_w, tile_stride_factor_h, tile_stride_factor_w, mask_pth, output_pth, lwst_level_idx, threshold)
        
        #For reproducibility ensure same seed
        random_seed = 2022
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        random.shuffle(self.all_image_tiles_hr)
        if self.mode=="train":
            self.image_dataset  = self.all_image_tiles_hr[0:int(0.8*len(self.all_image_tiles_hr))]
        else:
            self.image_dataset  = self.all_image_tiles_hr[int(0.8*len(self.all_image_tiles_hr)):]

        del self.all_image_tiles_hr

        # compute otf of blur kernel
        cFT = psf2otf(blur_kernel, (tile_h, tile_w))
        # this is our forward image formation model as a function
        self.Afun = lambda x: np.real(ifft2(fft2(x) * cFT))  

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        img =  self.image_dataset[index]
        #Generate blurred images
        gen_image,label = self.generate_blur(img)
        if self.transforms is not None:
            trans_img = self.transforms(Image.fromarray((gen_image*255).astype(np.uint8)))[0]
            return torch.stack((trans_img,trans_img,trans_img)), label
        else:
            return gen_image, label

    def generate_blur(self,img):
        """
        For adding artificial ink stains on a given image
        1: Blur free
        0: With blur
        """
        #For classification
        p = torch.rand(1).item()
        img = img/255
        if p<0.5: #50% chance for clean and ink stained data
            label = 1
            noise_img = img
        else:
            noise_img = self.blur_process(img)
            label = 0
        
        return noise_img, label

    def blur_process(self,img):
        # simulated measurements for all 3 color channels and noise
        p = torch.rand(1).item()
        if p<0.4:
            sig = 0
        else: #Produces with varying amounts of noise
            sig = self.sigma*torch.rand(1)
        b = np.zeros(img.shape)
        for it in range(3):
            b[:, :, it] = self.Afun(img[:, :, it]) + (sig*torch.normal(mean=0, std=1,size=(img.shape[0], img.shape[1]))).numpy()
        return b 
