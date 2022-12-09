# Investigating different priors for image deconvolution in histopathology images
In this course project, I have explored different priors for the non blind image deconvolution using adam optimizer in histopathology images. Overall I have explore 9 different priors in this project
1. Anisotropic, Isotropic Total variation
2. L1, L2
3. Hessian Schatten
4. Laplacian
5. Maximize cells
6. Cross Entropy, KL divergence

<img src="https://github.com/Vishwesh4/PriorsDeconvolution/blob/master/images/deconv.png" align="center" width="500"><figcaption>Fig.1 - Non blind deconvolution using different priors on histopathology</figcaption></a>

## Getting Started

### Prerequisites 

Python packages required
```
numpy
tqdm
torch
torchvision
matplotlib
wandb
pypher
torchmetrics
trainer
```
trainer package can be installed from this [link](https://github.com/Vishwesh4/TrainerCode)
### Dataset
For the experiments, I have used a single slide from the pubically available TIGER dataset(link)[https://tiger.grand-challenge.org/Data/]. The slide used for the experiments is 104S.tif. Do ensure the slide image is stored in `dataset` and its corresponding mask file is stored in `dataset/masks`

## File Descriptions
- `helper/CannyEdgePytorch`: The code is taken from DCurro's repository[link](https://github.com/DCurro/CannyEdgePytorch), for differentiable canny edge implementation
- `helper/deconv_adam.py`: Code for deconvolution using adam optimizer
- `helper/extract_patches.py`: Code for extracting patches given a whole slide image and mask
- `helper/priors.py`: Code with all priors implementation
- `training`: Directory for training model for cross entropy and kl divergence. To run training use the command 
```
python train.py - c CONFIG_LOCATION.yml
```
- `training/utils/trainutils.py`: File with all helper functions for loading dataset and model
- `get_image.py`: Code for running all the priors for a single image. Run using the following command
```
python get_image.py -b {0,1,2} -l {0,1} -n {0,1}
```
- `get_metrics.py`: Code for running all the priors for test set and generate all the image metrics statistics. Run it using the command
```
python get_metrics.py -b {0,1,2} -l {0,1} -n {0,1}
```
- `run_experiments.sh`: For runnning all the set of experiments using different blur kernel parameters. Use this to generate table given in the report 

