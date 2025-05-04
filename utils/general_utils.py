#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
from PIL import Image
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    """
    Converts a PIL image to a PyTorch tensor.
    Added fix for potential ValueError during resize.
    """
    try:
        # Ensure input is a PIL Image object if it isn't already
        if not isinstance(pil_image, Image.Image):
             # Try converting from numpy array if possible
             if isinstance(pil_image, np.ndarray):
                 pil_image_obj = Image.fromarray(pil_image)
             else:
                 raise TypeError(f"Input to PILtoTorch wasn't PIL Image or Numpy Array, got {type(pil_image)}")
        else:
            pil_image_obj = pil_image

        # --- Fix Start ---
        # Convert PIL Image to NumPy array, ensure it's contiguous, then back to PIL
        np_image = np.array(pil_image_obj)
        contiguous_np_image = np.ascontiguousarray(np_image)
        pil_image_copy = Image.fromarray(contiguous_np_image)
        # --- Fix End ---

        # Now resize the guaranteed copy
        # NOTE: PIL resize takes (width, height) tuple
        # Ensure 'resolution' is in the correct format (W, H tuple)
        if isinstance(resolution, (int, float)):
             # If resolution is a single number (like scale factor from -r arg),
             # calculate target W, H. Assuming it's a scaling factor here.
             # This might need adjustment based on how 'resolution' is passed.
             # Original code passed 'resolution' directly, which might imply
             # it expected a tuple or worked differently. Let's assume it needs
             # the target size tuple (W, H).
             # If 'resolution' was meant to be the downscale factor (e.g. 4 from -r 4)
             orig_w, orig_h = pil_image_copy.size
             target_w = orig_w // resolution
             target_h = orig_h // resolution
             resolution_tuple = (target_w, target_h)
        elif isinstance(resolution, (tuple, list)) and len(resolution) == 2:
             resolution_tuple = tuple(resolution) # Assume (W, H)
        else:
             # Default or fallback? Let's use original size if unclear
             print(f"Warning: Unexpected resolution format {resolution} in PILtoTorch. Using original size.")
             resolution_tuple = pil_image_copy.size


        # Use OpenCV resize for better NumPy handling
        # cv2 resize takes (width, height) tuple = resolution_tuple
        # interpolation can be cv2.INTER_AREA for shrinking, cv2.INTER_LINEAR for general
        interpolation_mode = cv2.INTER_AREA if resolution_tuple[0] < contiguous_np_image.shape[1] else cv2.INTER_LINEAR
        resized_np_array = cv2.resize(contiguous_np_image, resolution_tuple, interpolation=interpolation_mode)
        # resized_image_PIL = Image.fromarray(resized_np_array) # PIL version not needed now

        # Convert the resized numpy array to a torch tensor
        # resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
        resized_image = torch.from_numpy(resized_np_array).float() / 255.0 # Use float()

    except ValueError as e:
         # If resize still fails, print more info
         print(f"ERROR during resize in PILtoTorch: {e}")
         print(f"Input image type: {type(pil_image)}, Mode: {pil_image.mode if hasattr(pil_image, 'mode') else 'N/A'}, Size: {pil_image.size if hasattr(pil_image, 'size') else 'N/A'}")
         print(f"Target resolution tuple: {resolution_tuple}")
         raise e # Re-raise the error after printing info

    # Permute dimensions for PyTorch (C, H, W)
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    elif len(resized_image.shape) == 2: # Handle grayscale
        return resized_image.unsqueeze(dim=0) # Add channel dim -> (1, H, W)
    else:
         # This case was likely for alpha masks, might need adjustment
         print(f"Warning: Unexpected image shape {resized_image.shape} after resize in PILtoTorch")
         return resized_image.unsqueeze(dim=-1).permute(2, 0, 1) # Original logic


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
