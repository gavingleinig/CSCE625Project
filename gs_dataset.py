import os
import torch
import numpy as np
import random
from typing import List, Union
from PIL import Image
from argparse import ArgumentParser
from torch.utils.data import Dataset
import cv2
import glob
import re
import time
import traceback

from scene.cameras import Camera

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat 
from utils.graphics_utils import focal2fov



imagenet_templates_small = [
    'a photo of a {}',
    'a depiction of a small {}',
    # not used, refer to GaussianObject
]

class GSCachedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        hint_dir: str,
        image_size: int = 512,
        prompt: str = '',
        bg_white: bool = False,
        use_prompt_list: bool = False,
        sparse_train_file: str = None
    ):
        super().__init__()
        print("[INFO] Initializing GSCachedDataset (preload variant - loading ALL available hints)...")
        start_time = time.time()

        self.data_dir = data_dir
        self.hint_dir = hint_dir
        self.image_size = image_size
        self.prompt = prompt
        self.bg_white = bg_white
        self.use_prompt_list = use_prompt_list
        self.bg_value = 255 if self.bg_white else 0

        # Mapping from COLMAP ID to the hint_view index used in filenames
        self.camera_id_to_hint_view_map = {
            33: 3,
            28: 2,
            13: 1,
             0: 0,
        }
        print(f"[INFO] Using predefined map for {len(self.camera_id_to_hint_view_map)} camera IDs to hint views.")

        self.view_data = []

        # Load Sparse Training Image List
        self.sparse_image_names = None
        if sparse_train_file and os.path.exists(sparse_train_file):
            print(f"[INFO] Loading sparse training image list from: {sparse_train_file}")
            try:
                with open(sparse_train_file, 'r') as f:
                    self.sparse_image_names = {line.strip() for line in f if line.strip()}
                print(f"[INFO] Loaded {len(self.sparse_image_names)} image names for sparse training.")
            except Exception as e:
                print(f"[WARNING] Failed to read sparse train file {sparse_train_file}: {e}. Loading all cameras.")
                self.sparse_image_names = None
        elif sparse_train_file:
            print(f"[WARNING] Sparse train file specified but not found: {sparse_train_file}. Loading all cameras.")
        else:
             print("[INFO] No sparse training file specified. Attempting to load all cameras from COLMAP.")


        #Load Camera Parameters and Images
        try:
            print(f"[INFO] Loading cameras and target images from {self.data_dir}...")
            cam_extr_file = os.path.join(self.data_dir, "sparse/0/images.txt")
            images_dir = os.path.join(self.data_dir, "images")

            if not os.path.exists(cam_extr_file): raise FileNotFoundError(f"COLMAP images.txt not found at {cam_extr_file}")
            if not os.path.exists(images_dir): raise FileNotFoundError(f"Target image directory not found at {images_dir}")
            if not os.path.isdir(images_dir): raise NotADirectoryError(f"Target image path is not a directory: {images_dir}")

            cam_extrinsics = read_extrinsics_text(cam_extr_file)

            # Filtering Logic (Sparse file and Hint Map availability)
            filtered_extr_keys = list(cam_extrinsics.keys())
            # Filter by sparse file (if provided)
            if self.sparse_image_names is not None:
                original_count = len(filtered_extr_keys)
                filtered_extr_keys = [img_id for img_id in filtered_extr_keys if cam_extrinsics[img_id].name in self.sparse_image_names]
                print(f"[INFO] Filtered cameras based on sparse file: {original_count} -> {len(filtered_extr_keys)} cameras.")
            # Filter by hint map availability
            original_count_before_map_filter = len(filtered_extr_keys)
            mapped_filtered_extr_keys = [
                img_id for img_id in filtered_extr_keys
                if img_id in self.camera_id_to_hint_view_map
            ]
            if len(mapped_filtered_extr_keys) < original_count_before_map_filter:
                print(f"[INFO] Filtered cameras based on hint map availability: {original_count_before_map_filter} -> {len(mapped_filtered_extr_keys)} cameras.")

            filtered_extr_keys = mapped_filtered_extr_keys
            # End Filtering 

            if not filtered_extr_keys:
                 print("[WARNING] No camera IDs left after filtering. Cannot load any views.")

            #Get list of files in target images directory
            try:
                image_names_in_dir = {f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
                if not image_names_in_dir:
                    print(f"[WARNING] No image files (.png, .jpg, .jpeg) found in the target directory: {images_dir}")
            except Exception as e:
                 raise FileNotFoundError(f"Error listing files in target image directory {images_dir}: {e}")

            print(f"[INFO] Attempting to load targets and find/load ALL available hints for {len(filtered_extr_keys)} views...")
            views_loaded_count = 0
            views_skipped_no_target = 0
            views_skipped_target_load_error = 0
            views_skipped_no_hints_found = 0
            views_skipped_hint_load_error = 0 # Counts views where *all* hints failed load
            views_skipped_not_in_map = 0
            total_hints_loaded_across_all_views = 0
            individual_hint_load_errors = 0

            # Iterate through filtered camera views
            for i, img_id in enumerate(sorted(filtered_extr_keys)):
                extr = cam_extrinsics[img_id]
                camera_uid = extr.id
                image_name = extr.name
                image_path = os.path.join(images_dir, image_name)

                if camera_uid not in self.camera_id_to_hint_view_map:
                    print(f"[WARNING] Internal Warning: Camera UID {camera_uid} (img_id {img_id}, name {image_name}) not found in map during loop. Skipping.")
                    views_skipped_not_in_map += 1
                    continue

                # 1. Load Target Image
                target_image_np = None
                if image_name in image_names_in_dir:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        target_image_np = np.array(image)
                    except Exception as e:
                        print(f"[WARNING] Failed to load target image {image_path} for view UID {camera_uid}: {e}")
                        views_skipped_target_load_error += 1
                        continue # Skip this view if target fails
                else:
                    print(f"[WARNING] Target image '{image_name}' for view UID {camera_uid} not found in directory {images_dir}.")
                    views_skipped_no_target += 1
                    continue

                # 2. Find and Load ALL Available Hint Images for this view
                hint_images_np = [] 
                hint_view_index = self.camera_id_to_hint_view_map[camera_uid]
                hint_subdir = os.path.join(self.hint_dir, str(hint_view_index))

                if not os.path.isdir(hint_subdir):
                    print(f"[WARNING] Hint subdirectory does not exist for view UID {camera_uid}: {hint_subdir}. No hints can be loaded.")
                    views_skipped_no_hints_found += 1
                    continue 

                hint_pattern = os.path.join(hint_subdir, f"hint_view_{hint_view_index}_iter_*.png")
                hint_files = sorted(glob.glob(hint_pattern)) 

                if not hint_files:
                    print(f"[WARNING] No hint files found matching pattern for view UID {camera_uid}: {hint_pattern}")
                    views_skipped_no_hints_found += 1
                    continue # Skip this view if no hints found
                else:
                    # Try loading each found hint file
                    for hint_path in hint_files:
                        try:
                            hint_image = Image.open(hint_path).convert("RGB")
                            hint_images_np.append(np.array(hint_image))
                        except Exception as e:
                            print(f"[WARNING] Failed to load specific hint file {hint_path} for view UID {camera_uid}: {e}")
                            individual_hint_load_errors += 1

                # 3. Store if Target loaded AND at least one Hint was successfully loaded
                if target_image_np is not None and len(hint_images_np) > 0:
                    self.view_data.append({
                        'uid': camera_uid,
                        'target_image': target_image_np,
                        'hint_images': hint_images_np, 
                        'image_name': image_name
                    })
                    views_loaded_count += 1
                    total_hints_loaded_across_all_views += len(hint_images_np)
                elif target_image_np is not None: 
                     print(f"[WARNING] Target loaded for view UID {camera_uid}, but failed to load any of the {len(hint_files)} found hint files.")
                     views_skipped_hint_load_error += 1



        except Exception as e:
            print(f"[ERROR] Error during dataset initialization: {e}")
            traceback.print_exc()
            raise

        print(f"\n[INFO] Loop finished. Processed {len(filtered_extr_keys)} filtered camera IDs.")
        if not self.view_data:
            print("[ERROR] self.view_data is empty after processing all views.")
            raise ValueError("No valid camera/image/hint pairs loaded. Check [WARNING]/[ERROR] logs.")

        end_time = time.time()
        print(f"\n[INFO] Finished dataset initialization ({end_time - start_time:.2f}s).")
        print(f" Summary:")
        print(f"   Views loaded successfully (target + >=1 hint): {views_loaded_count}")
        print(f"   Views skipped (target not found in dir list): {views_skipped_no_target}")
        print(f"   Views skipped (target load error): {views_skipped_target_load_error}")
        print(f"   Views skipped (no hint files found/subdir missing): {views_skipped_no_hints_found}")
        print(f"   Views skipped (hints found but all failed load): {views_skipped_hint_load_error}")
        if views_skipped_not_in_map > 0: print(f"   Views skipped (not in hint map - internal warning): {views_skipped_not_in_map}")
        print(f"   Total individual hint images loaded across all views: {total_hints_loaded_across_all_views}")
        if individual_hint_load_errors > 0: print(f"   Total individual hint image load errors: {individual_hint_load_errors}")


    def __len__(self):
        return len(self.view_data)

    def resize_larger_intermediate(self, image_np: np.ndarray, target_min_size: int) -> np.ndarray:
        h, w, _ = image_np.shape

        if h == 0 or w == 0:
             print(f"[WARNING] resize_larger_intermediate received zero-dimension image shape: {(h, w)}")
             # Return a small black image or raise error?
             return np.zeros((target_min_size, target_min_size, 3), dtype=image_np.dtype)

        min_dim = min(h, w)
        if min_dim == 0: # Avoid division by zero if one dim is zero but not both
             print(f"[WARNING] resize_larger_intermediate received image with one zero dimension: {(h, w)}")
             return np.zeros((target_min_size, target_min_size, 3), dtype=image_np.dtype)

        if min_dim == target_min_size:
            return image_np # Already the correct minimum size

        scale = target_min_size / min_dim
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Ensure new dimensions are at least 1 pixel
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        try:
            # Use INTER_LINEAR as a general purpose scaler (good for up/down)
            resized_img = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized_img
        except Exception as e:
            print(f"[ERROR] Failed during resize_larger_intermediate (shape: {image_np.shape}, scale: {scale:.2f}, new_w: {new_w}, new_h: {new_h}): {e}")
            traceback.print_exc()
            # Return a blank image of target size or raise?
            return np.zeros((target_min_size, target_min_size, 3), dtype=image_np.dtype)

    def __getitem__(self, idx):
        view_item = self.view_data[idx]
        target_image_np_raw = view_item['target_image']
        available_hints_raw = view_item['hint_images']
        if not available_hints_raw:
             # This should ideally not happen due to checks in __init__
             raise ValueError(f"Internal Error: No raw hints available for view UID {view_item['uid']} at index {idx}")
        hint_image_np_raw = random.choice(available_hints_raw)

        #Resize both images to intermediate size 
        try:
            hint_intermediate = self.resize_larger_intermediate(hint_image_np_raw, self.image_size)
            target_intermediate = self.resize_larger_intermediate(target_image_np_raw, self.image_size)

            # Verify dimensions after resize are identical
            if hint_intermediate.shape != target_intermediate.shape:
                 print(f"[WARNING] Intermediate shapes mismatch after resize for index {idx} (UID {view_item['uid']}):")
                 print(f"  Hint shape: {hint_intermediate.shape}, Target shape: {target_intermediate.shape}")
                 # Attempt to resize target to match hint as a fallback
                 try:
                     target_intermediate = cv2.resize(target_intermediate, (hint_intermediate.shape[1], hint_intermediate.shape[0]), interpolation=cv2.INTER_LINEAR)
                     print(f"  Attempted corrective resize of target to: {target_intermediate.shape}")
                     if hint_intermediate.shape != target_intermediate.shape:
                        raise ValueError("Corrective resize failed to match shapes.")
                 except Exception as resize_err:
                    print(f"[ERROR] Corrective resize failed: {resize_err}")
                    raise ValueError(f"Unrecoverable shape mismatch after resize for index {idx}")


        except Exception as e:
             print(f"[ERROR] Failed during resize_larger_intermediate call in __getitem__ for index {idx} (UID {view_item['uid']}): {e}")
             traceback.print_exc()
             dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
             return {'jpg': dummy_img, 'txt': f'error_resize_{idx}', 'hint': dummy_img}

        #Perform Linked Random Crop
        H_inter, W_inter, _ = hint_intermediate.shape
        crop_h = self.image_size
        crop_w = self.image_size

        max_start_h = H_inter - crop_h
        max_start_w = W_inter - crop_w
        start_h = random.randint(0, max_start_h) if max_start_h >= 0 else 0
        start_w = random.randint(0, max_start_w) if max_start_w >= 0 else 0

        try:
            hint_cropped = hint_intermediate[
                start_h : start_h + crop_h,
                start_w : start_w + crop_w,
                :
            ]
            target_cropped = target_intermediate[
                start_h : start_h + crop_h,
                start_w : start_w + crop_w,
                :
            ]

        except Exception as e:
            print(f"[ERROR] Failed during cropping stage in __getitem__ for index {idx} (UID {view_item['uid']}): {e}")
            print(f"  Intermediate Shape: {(H_inter, W_inter)}, Crop Window: start=({start_h},{start_w}), size=({crop_h},{crop_w})")
            traceback.print_exc()
            dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            return {'jpg': dummy_img, 'txt': f'error_crop_{idx}', 'hint': dummy_img}


        # Normalize and Convert to Float32
        # Hint ('hint'): Normalize to [0, 1]
        hint_output = hint_cropped.astype(np.float32) / 255.0
        # Target ('jpg'): Normalize to [-1, 1]
        target_output = target_cropped.astype(np.float32) / 127.5 - 1.0

        #Get Prompt 
        if self.use_prompt_list and imagenet_templates_small:
             prompt_text = random.choice(imagenet_templates_small).format(self.prompt)
        else:
             prompt_text = f'a photo of a {self.prompt}' if self.prompt else 'a photo'

        return {
            'jpg': target_output,  # HWC numpy float32 [-1, 1]
            'txt': prompt_text,
            'hint': hint_output   # HWC numpy float32 [0, 1]
        }