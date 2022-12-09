import sys
import os
from os.path import exists
import glob
import random
import warnings

import cv2
import numpy as np
import openslide
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
from pathlib import Path

class ExtractPatches(Dataset):
    """
    WSI dataset, This class based on given image path, extract points at a uniform stride belonging inside the whole slide
    """

    def __init__(
        self,
        image_pth,
        tile_h,
        tile_w,
        tile_stride_factor_h,
        tile_stride_factor_w,
        mask_pth=None,
        output_pth=None,
        lwst_level_idx=0,
        threshold=0.7,
        transform=None,
        **kwargs
    ):

        """
        Args:
            image_pth (str): path to wsi.
            tile_h (int): tile height
            tile_w (int): tile width
            tile_stride_factor_h (int): stride height factor, height will be tile_height * factor
            tile_stride_factor_w (int): stride width factor, width will be tile_width * factor
            spacing(float): Specify this value if you want to extract patches at a given spacing
            mask_pth(str): Directory where all masks are stored, if none is given then masks are extracted automatically
            output_pth(str): Directory where all the masks and template if calculated are stored
            mode (str): train or val, split the slides into trainset and val set
            train_split(float): Between 0-1, ratio of split between train and val set
            lwst_level_idx (int): lowest level for patch indexing
            threshold(float): For filtering from mask
        """

        self.image_path = image_pth
        self.output_path = output_pth
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_stride_h = int(tile_h*tile_stride_factor_h)
        self.tile_stride_w = int(tile_w*tile_stride_factor_w)
        self.hr_level = lwst_level_idx
        self.mask_path = mask_pth
        self.transform = transform
        self.threshold = threshold

        for key,value in kwargs.items():
            setattr(self,key,value)

        #Get all mask paths if applicable
        if self.mask_path is not None:
            temppth = Path(self.mask_path)
            if temppth.is_dir():
                self.all_masks = list(temppth.glob("*"))
            else:
                print(f"Found {len(self.all_masks)} masks")
                self.all_masks = list(self.mask_path)

        #Load all extracted patches into RAM
        self.all_image_tiles_hr = self.tiles_array()

        print(f"Extracted {len(self.all_image_tiles_hr)} patches")

    def __len__(self):
        return len(self.all_image_tiles_hr)

    def __getitem__(self, index):
        img = self.all_image_tiles_hr[index]
        if self.transform is not None:
            return self.transform(img)
        else:
            return img

    def tiles_array(self):

        all_wsipaths = []
        if isinstance(self.image_path,list):
            #Check if images in list exist
            for i in range(len(self.image_path)):
                if not(exists(self.image_path[i])):
                    raise Exception("WSI file does not exist in: %s" % str(self.image_path[i]))
            all_wsipaths = self.image_path.copy()
        else:
            # Check image
            if not exists(self.image_path):
                raise Exception("WSI file does not exist in: %s" % str(self.image_path))
            if Path(self.image_path).suffix[1:] in ["tif","svs"]:
                all_wsipaths.append(self.image_path)
            for file_ext in ['tif', 'svs']:
                all_wsipaths = all_wsipaths + glob.glob('{}/*.{}'.format(self.image_path, file_ext))
        random.shuffle(all_wsipaths)
        
        #Select subset of slides for training/val setup
        wsipaths = all_wsipaths

        with tqdm(enumerate(sorted(wsipaths))) as t:

            all_image_tiles_hr = []

            for wj, wsipath in t:
                t.set_description(
                    "Loading wsis.. {:d}/{:d}".format(1 + wj, len(wsipaths))
                )

                "generate tiles for this wsi"
                image_tiles_hr = self.get_wsi_patches(wsipath)

                # Check if patches are generated or not for a wsi
                if len(image_tiles_hr) == 0:
                    print("bad wsi, no patches are generated for", str(wsipath))
                    continue
                else:
                    all_image_tiles_hr.append(image_tiles_hr)

            # Stack all patches across images
            all_image_tiles_hr = np.concatenate(all_image_tiles_hr)
        
        return all_image_tiles_hr
    
    def _get_mask(self, wsipth):
        filename, file_extension = os.path.splitext(Path(wsipth).name)
        indv_mask_pth = list(filter(lambda x: filename in str(x),self.all_masks))
        mask = io.imread(str(indv_mask_pth[0]))
        return mask

    def _getpatch(self, scan, x, y):

        'read low res. image'
        'hr patch'
        image_tile_hr = scan.read_region((x, y), self.hr_level, (self.tile_w, self.tile_h)).convert('RGB')
        image_tile_hr = np.array(image_tile_hr).astype('uint8')

        return image_tile_hr

    def _get_slide(self, wsipth):
        """
        Returns openslide object
        """
        scan = openslide.OpenSlide(wsipth)
        return scan

    def get_wsi_patches(self, wsipth):
        "read the wsi scan"
        scan = self._get_slide(wsipth)
        mask = self._get_mask(wsipth)
        # scan = openslide.OpenSlide(wsipth)

        "downsample multiplier"
        """
        due to the way pyramid images are stored,
        it's best to use the lower resolution to
        specify the coordinates then pick high res.
        from that (because low. res. pts will always
        be on high res image but when high res coords
        are downsampled, you might lose that (x,y) point)
        """

        iw, ih = scan.dimensions
        sh, sw = self.tile_stride_h, self.tile_stride_w
        ph, pw = self.tile_h, self.tile_w

        patch_id = 0
        image_tiles_hr = []
        
        for y,ypos in enumerate(range(sh, ih - 1 - ph, sh)):
            for x,xpos in enumerate(range(sw, iw - 1 - pw, sw)):
                if self._isforeground((xpos, ypos), mask):  # Select valid foreground patch
                    # coords.append((xpos,ypos))
                    image_tile_hr = self._getpatch(scan, xpos, ypos)

                    image_tiles_hr.append(image_tile_hr)
                    
                    patch_id = patch_id + 1

        # Concatenate
        if len(image_tiles_hr) == 0:
            image_tiles_hr == []
        else:
            image_tiles_hr = np.stack(image_tiles_hr, axis=0).astype("uint8")
        
        return image_tiles_hr

    def _isforeground(self, coords, mask):
        x,y = coords
        patch = mask[y:y+self.tile_w,x:x+self.tile_w]
        return (np.sum(patch)/float(self.tile_w*self.tile_h))>=self.threshold