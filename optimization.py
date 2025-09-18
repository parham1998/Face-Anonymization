# =============================================================================
# Import required libraries
# =============================================================================
import os
import numpy as np
from scipy.ndimage import binary_dilation

import torch
from torchvision.utils import save_image

from FaceParsing.interface import FaceParsing

from utils import *


@torch.enable_grad()
class Optimization:
    def __init__(self, args, sampler, initialize_seed):
        self.device = args.device
        self.dataloader = args.dataloader
        #
        self.sampler = sampler
        #
        self.initialize_seed = initialize_seed
        #
        self.source_dir = args.source_dir
        self.target_path = args.target_path
        self.anonymized_image_dir = args.anonymized_image_dir
        #
        self.image_size = args.image_size
        self.diffusion_steps = args.diffusion_steps
        #
        self.sim_weight = args.sim_weight
        self.dissim_weight = args.dissim_weight
        self.optim_weight = args.optim_weight
        #
        self.masks = args.masks
        self.excluded_masks = args.excluded_masks

    def get_mask(self, pred, number):
        return pred == number

    def create_mask_with_exclusions(self, pred, all_labels, excluded_masks, dilation_kernel=7, dilate=True):
        all_labels = set(all_labels)
        target_labels = all_labels - set(excluded_masks)

        # Build initial mask to remove (mask out these labels)
        mask = None
        for label in target_labels:
            part_mask = (pred == label)
            mask = part_mask if mask is None else (mask | part_mask)

        # Dilate excluded regions (if needed)
        if dilate and excluded_masks:
            preserved_mask = None
            for label in excluded_masks:
                binary = (pred == label).cpu().numpy()
                dilated = binary_dilation(binary.squeeze(
                    0), structure=np.ones((dilation_kernel, dilation_kernel)))
                dilated = torch.from_numpy(dilated).to(pred.device)
                preserved_mask = dilated if preserved_mask is None else (
                    preserved_mask | dilated)
            # Subtract preserved regions from the mask
            mask = mask & (~preserved_mask)

        return mask.float()

    def run(self):
        timer = MyTimer()
        time_list = []
        result_dir = self.anonymized_image_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with torch.no_grad():
            for i, (fname, image, target_image) in enumerate(self.dataloader):
                self.initialize_seed(0)
                #
                timer.tic()
                #
                image_name = fname[0]
                image = image.to(self.device)
                target_image = target_image.to(self.device)
                B = image.shape[0]

                face_parsing = FaceParsing(self.device)
                pred = face_parsing(image)

                mask = None
                for x in self.masks:
                    if mask is not None:
                        mask |= self.get_mask(pred, x)
                    else:
                        mask = self.get_mask(pred, x)
                if self.excluded_masks is not []:
                    mask = self.create_mask_with_exclusions(pred,
                                                            self.masks,
                                                            self.excluded_masks,
                                                            dilation_kernel=7,
                                                            dilate=True)

                mask = mask.float().reshape(
                    B, 1, self.image_size, self.image_size).to(self.device)
                masked_image = image * (1 - mask)

                batch = {
                    "image": image,
                    "txt": B * [''],
                    "mask": mask,
                    "masked_image": masked_image,
                }

                c = self.sampler.model.cond_stage_model.encode(batch["txt"])
                c_cat = list()
                for ck in self.sampler.model.concat_keys:
                    cc = batch[ck].float()
                    if ck != self.sampler.model.masked_image_key:
                        bchw = [B, 4, self.image_size //
                                8, self.image_size // 8]
                        cc = torch.nn.functional.interpolate(
                            cc, size=bchw[-2:])
                    else:
                        cc = self.sampler.model.get_first_stage_encoding(
                            self.sampler.model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = self.sampler.model.get_unconditional_conditioning(
                    B, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [self.sampler.model.channels,
                         self.image_size // 8, self.image_size // 8]

                # start code
                _t = 999
                z = self.sampler.model.get_first_stage_encoding(
                    self.sampler.model.encode_first_stage(image))
                t = torch.tensor([_t] * B, device=self.device)
                z_t = self.sampler.model.q_sample(x_start=z, t=t)

                samples_cfg, _ = self.sampler.sample(
                    self.diffusion_steps,
                    B,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=0,
                    unconditional_conditioning=uc_full,
                    x_T=z_t,
                    _t=_t + 1,
                    log_every_t=1,
                    target_image=target_image,
                    sim_weight=self.sim_weight,
                    dissim_weight=self.dissim_weight,
                    optim_weight=self.optim_weight,
                    image=image)
                
                x_samples_ddim = self.sampler.model.decode_first_stage(samples_cfg)
                result = torch.clamp(x_samples_ddim, min=-1, max=1)
                #
                avg_time = timer.toc()
                time_list.append(avg_time)
                #
                os.makedirs(os.path.join(self.anonymized_image_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(self.anonymized_image_dir, 'masks'), exist_ok=True)
                print(i)
                for x in range(result.shape[0]):
                    save_image((result[x] + 1) / 2, os.path.join(self.anonymized_image_dir, 'images', f'{image_name}.png'))
                    save_image((masked_image[x] + 1) / 2, os.path.join(self.anonymized_image_dir, 'masks', f'{image_name}_m.png'))

        #
        print('Time: ', round(np.average(time_list), 2))
        result_fn = os.path.join(self.anonymized_image_dir, "time.txt")
        with open(result_fn, 'a') as f:
            f.write(f"Time: {round(np.average(time_list),2)}\n")
        f.close()
