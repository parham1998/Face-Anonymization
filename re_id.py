# =============================================================================
# Import required libraries
# =============================================================================
import os
import glob
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from utils import *


def Re_Identification_Rate(args):
    # Load test model
    test_model = load_FR_model(args)
    # False acceptance rate (FAR) is set to 0.01
    thr = 0.409131

    result_dir = args.anonymized_image_dir
    result_fn = os.path.join(result_dir, "result.txt")

    combined_dir = os.path.join(result_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    size = test_model['facenet'][0]
    model = test_model['facenet'][1]

    FAR001 = 0
    total = 0
    for img_path in glob.glob(os.path.join(result_dir, 'images', "*.png")):
        anon_image = read_img(img_path, 0.5, 0.5, args.device)
        #
        image_name = os.path.basename(img_path)
        real_image = read_img(os.path.join(
            args.source_dir, image_name), 0.5, 0.5, args.device)
        #
        if args.MTCNN_cropping:
            bb_src1 = alignment(Image.open(img_path).convert("RGB"))
            bb_src2 = alignment(Image.open(os.path.join(
                args.source_dir, image_name)).convert("RGB"))
    
            if bb_src1 is not None and bb_src2 is not None:
                anon_image_hold = anon_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                _, _, h, w = anon_image_hold.shape
                if h != 0 and w != 0:
                    anon_image = anon_image_hold
                        
                real_image_hold = real_image[:, :, round(bb_src2[1]):round(
                    bb_src2[3]), round(bb_src2[0]):round(bb_src2[2])]
                _, _, h, w = real_image_hold.shape
                if h != 0 and w != 0:
                    real_image = real_image_hold
        #
        anon_embbeding = model.forward(
            (F.interpolate(anon_image, size=size, mode='bilinear')))
        real_embbeding = model.forward(
            (F.interpolate(real_image, size=size, mode='bilinear')))
        cos_simi = torch.cosine_similarity(anon_embbeding, real_embbeding)
        if cos_simi.item() > thr:
            FAR001 += 1
        total += 1

        # Combine the protected and test images for visualization
        anon_image = cv2.imread(img_path)
        real_image = cv2.imread(os.path.join(args.source_dir, image_name))
        #
        combined_img = np.concatenate([real_image, anon_image], 1)
        combined_fn = f"{image_name.split('.')[0]}_cos_simi_{cos_simi.item():.4f}.png"
        cv2.imwrite(os.path.join(combined_dir, combined_fn), combined_img)

    result_str = f"Re_Id rate: {FAR001/total:.4f} \n"
    print(result_str)
    with open(result_fn, 'a') as f:
        f.write(result_str)
    f.close()
