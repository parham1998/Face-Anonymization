# =============================================================================
# Import required libraries
# =============================================================================
import os
import random
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from dataset import ImageDataset
from optimization import Optimization
from re_id import Re_Identification_Rate


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--source_dir',
                        default="assets/datasets/CelebA-HQ/female",
                        type=str,
                        help="source images folder path")
    parser.add_argument('--target_path',
                        default="assets/target_images/syn_tar_1_female.png",
                        type=str,
                        help="path to the target image")
    parser.add_argument('--anonymized_image_dir',
                        default="results",
                        type=str)

    parser.add_argument('--MTCNN_cropping',
                        default=True,
                        type=bool)

    parser.add_argument('--image_size',
                        default=512,
                        type=int)
    parser.add_argument('--diffusion_steps',
                        default=45,
                        type=int)

    parser.add_argument('--sim_weight',
                        default=1,
                        type=int)
    parser.add_argument('--dissim_weight',
                        default=0.0,
                        type=int)
    parser.add_argument('--optim_weight',
                        default=0.8,
                        type=int)
    
    '''
    label_list = ['1: skin', '2: nose', '3: eye_glasses', '4: l_eye', 
                  '5: r_eye', '6: l_brow', '7:r_brow', '10: mouth', 
                  '11: u_lip', '12: l_lip']
    '''
    parser.add_argument('--masks',
                        default=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
                        type=list,
                        help="")
    parser.add_argument('--excluded_masks',
                        default=[3],
                        type=list,
                        help="")
                
    args = parser.parse_args()
    return args


def initialize_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    #
    initialize_seed(0)
    #
    args = parse_args()
    #
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    config = OmegaConf.load(
        'configs/stable-diffusion/v2-inpainting-inference.yaml')
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(
        'assets/ldm/512-inpainting-ema.ckpt')["state_dict"], strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(args.device)
    sampler = DDIMSampler(model, args.device)

    # Load the dataset
    dataset = ImageDataset(
        args.source_dir,
        args.target_path,
        transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5])]
                           )
    )
    args.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    opt = Optimization(args, sampler, initialize_seed)
    opt.run()

    Re_Identification_Rate(args)
