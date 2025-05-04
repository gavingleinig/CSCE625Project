import os
import sys
import uuid
import torch
import numpy as np
import torchvision
import random
import time
from argparse import ArgumentParser, Namespace
from random import randint
from tqdm import tqdm
from functools import partial

# Imports from original GS training (traings.py)
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, network_gui
from scene import GaussianModel, Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import update_pose
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Imports for ControlNet/LoRA loading
from torch import nn
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from minlora import add_lora, LoRAParametrization
from PIL import Image
import torchvision.transforms as transforms

# Imports for Stage 2 logic
from utils.graphics_utils import interpolate_camera_poses # *** IMPORTANT: Ensure this function is available in your utils ***
from utils.diff_utils import process
import lpips

def run_controlnet_inference(controlnet_model, ddim_sampler, gs_render_np, prompt_text, args):
    controlnet_outputs, _ = process(
        model=controlnet_model,
        ddim_sampler=ddim_sampler,
        input_image=gs_render_np,
        prompt=args.prompt,
        a_prompt='best quality, sharp',
        n_prompt='blur, lowres, bad anatomy, worst quality',
        num_samples=1,
        image_resolution=min(gs_render_np.shape[0], gs_render_np.shape[1]),
        ddim_steps=args.ddim_steps,
        guess_mode=False,
        strength=args.control_strength,
        scale=args.guidance_scale,
        eta=args.ddim_eta,
        denoise_strength=0.2
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", f"{unique_str[0:10]}_stage2")

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args_stage2"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.model_path, "logs"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_refine(dataset, opt, pipe, test_iterations, save_iterations,
                      checkpoint_iterations, start_checkpoint, debug_from, args):
    print("Starting Stage 2 Refinement Training...")
    tb_writer = prepare_output_and_logger(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, extra_opts=args)
    gaussians.training_setup(opt)

    first_iter = 0
    if start_checkpoint:
        print(f"Loading GS checkpoint: {start_checkpoint}")
        (model_params, first_iter) = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Resuming GS training from iteration {first_iter}")
    else:
        print("[ERROR] No GS start_checkpoint provided for Stage 2!")
        sys.exit(1)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Loading ControlNet+LoRA model...")
    controlnet_model = create_model(f'./models/{args.controlnet_model_name}.yaml').cpu()
    controlnet_model.load_state_dict(load_state_dict(args.sd_v15_path, location='cpu'), strict=False)
    controlnet_model.load_state_dict(load_state_dict(f'./models/{args.controlnet_model_name}.pth', location='cpu'), strict=False)
    controlnet_model.learning_rate = 0
    controlnet_model.sd_locked = True
    controlnet_model.only_mid_control = False
    controlnet_model.train_lora = True # Important for LoRA weight application

    lora_config = {
        nn.Embedding: {"weight": partial(LoRAParametrization.from_embedding, rank=args.lora_rank)},
        nn.Linear: {"weight": partial(LoRAParametrization.from_linear, rank=args.lora_rank)},
        nn.Conv2d: {"weight": partial(LoRAParametrization.from_conv2d, rank=args.lora_rank)}
    }
    for name, module in controlnet_model.model.diffusion_model.named_modules():
        if name.endswith('transformer_blocks'): add_lora(module, lora_config=lora_config)
    for name, module in controlnet_model.control_model.named_modules():
        if name.endswith('transformer_blocks'): add_lora(module, lora_config=lora_config)
    add_lora(controlnet_model.cond_stage_model, lora_config=lora_config)

    print(f"Loading LoRA checkpoint: {args.lora_checkpoint_path}")
    if not os.path.exists(args.lora_checkpoint_path):
            print(f"[ERROR] LoRA checkpoint not found at {args.lora_checkpoint_path}")
            sys.exit(1)
    controlnet_model.load_state_dict(load_state_dict(args.lora_checkpoint_path, location='cuda'), strict=False)
    controlnet_model = controlnet_model.cuda().eval()
    ddim_sampler = DDIMSampler(controlnet_model)
    print("ControlNet+LoRA model loaded.")

    loss_fn_lpips = lpips.LPIPS(net='alex').cuda()

    camera_id_to_hint_view_map = {
        33: 3,
        28: 2,
        13: 1,
        0: 0,
    }

    print("Loading all training cameras from the scene object...")
    all_train_cameras = scene.getTrainCameras()
    if not all_train_cameras:
        print("[ERROR] No training cameras found in the scene object.")
        sys.exit(1)
    print(f"Found {len(all_train_cameras)} total training cameras initially.")

    print("[INFO] Applying hint map filter to all loaded cameras...")
    original_count = len(all_train_cameras)

    expected_attribute_name = 'uid'

    cameras_after_hint_map_filter = []
    if all_train_cameras:
        first_camera = all_train_cameras[0]
        if not hasattr(first_camera, expected_attribute_name):
            print(f"[ERROR] Camera objects seem to lack the '{expected_attribute_name}' attribute needed for hint map filtering. Cannot proceed.")
            sys.exit(1)
        else:
            cameras_after_hint_map_filter = [
                cam for cam in all_train_cameras
                # Ensure attribute exists before accessing, then check map
                if hasattr(cam, expected_attribute_name) and getattr(cam, expected_attribute_name) in camera_id_to_hint_view_map
            ]
            print(f"[INFO] Filtered based on hint map: {original_count} -> {len(cameras_after_hint_map_filter)} cameras.")


    else: # Initial list from scene was empty
        print("[INFO] No cameras were loaded initially, skipping hint map filter.")


    camera_input_for_interpolation = cameras_after_hint_map_filter
    print(f"[INFO] Final number of cameras selected for interpolation (hint map filter only): {len(camera_input_for_interpolation)}")


    if not camera_input_for_interpolation:
        print("[ERROR] No cameras remain after applying the hint map filter. Cannot generate virtual poses.")
        if original_count > 0 and hasattr(all_train_cameras[0], expected_attribute_name):
            found_ids = {getattr(cam, expected_attribute_name) for cam in all_train_cameras if hasattr(cam, expected_attribute_name)}
        sys.exit(1)


    print(f"Generating virtual camera poses using {len(camera_input_for_interpolation)} cameras as input...")
    try:
        # Assume args.num_virtual_views exists or replace with the correct variable/value
        virtual_Rs, virtual_Ts, _, _, _ = interpolate_camera_poses(camera_input_for_interpolation, args.num_virtual_views)

        print(f"Generated {len(virtual_Rs)} virtual poses.")
        if not virtual_Rs:
             raise ValueError("Interpolation function returned no poses, even with valid input cameras.")

    except ValueError as ve:
        print(f"[ERROR] {ve}")
        sys.exit(1)
    except AttributeError as ae:
        print(f"[ERROR] An attribute error occurred during interpolation: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to generate virtual poses due to an unexpected error: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


    print("Pre-generating ControlNet targets for virtual views...")
    controlnet_targets = []
    to_tensor = transforms.ToTensor()

    if not camera_input_for_interpolation:
        print("[ERROR] Internal Error: camera_input_for_interpolation is empty, cannot get sample parameters.")
        sys.exit(1)

    sample_cam = camera_input_for_interpolation[0]
    try:
        img_H, img_W = sample_cam.image_height, sample_cam.image_width
        sample_FoVx, sample_FoVy = sample_cam.FoVx, sample_cam.FoVy

    except AttributeError as e:
         print(f"[ERROR] Sample camera object (type: {type(sample_cam)}) is missing required attributes (e.g., image_height, image_width, FoVx, FoVy). Error: {e}")
         print("[INFO] Check the attributes of the Camera objects returned by scene.getTrainCameras().")
         sys.exit(1)

    for idx in tqdm(range(len(virtual_Rs)), desc="Generating ControlNet Targets"):
        try:
            virtual_cam = Camera(colmap_id=idx, R=virtual_Rs[idx], T=virtual_Ts[idx],
                                FoVx=sample_cam.FoVx, FoVy=sample_cam.FoVy,
                                image=torch.zeros(3, img_H, img_W), gt_alpha_mask=None,
                                image_name=f"virtual_{idx}", uid=-idx-1,
                                data_device="cuda", mono_depth=None)
        except Exception as e:
            print(f"\n[ERROR] Failed to create virtual Camera object: {e}")
            print("[INFO] Check Camera class constructor and required arguments.")
            sys.exit(1)

        with torch.no_grad():
            render_pkg_virtual = render(virtual_cam, gaussians, pipe, background)
            rendered_image_gs = render_pkg_virtual["render"].clamp(0.0, 1.0)

        gs_render_np = (rendered_image_gs.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)


        try:
            return_value_from_process = process(
                model=controlnet_model,
                ddim_sampler=ddim_sampler,
                input_image=gs_render_np,
                prompt=args.prompt,
                a_prompt='best quality, sharp',
                n_prompt='blur, lowres, bad anatomy, worst quality',
                num_samples=1,
                image_resolution=min(gs_render_np.shape[0], gs_render_np.shape[1]),
                ddim_steps=args.ddim_steps,
                guess_mode=False,
                strength=args.control_strength,
                scale=args.guidance_scale,
                eta=args.ddim_eta,
                denoise_strength=0.2
            )


            controlnet_outputs = None
            if isinstance(return_value_from_process, (tuple, list)) and len(return_value_from_process) >= 1:
                controlnet_outputs = return_value_from_process[0] 
            else:
                print(f"WARN: process returned unexpected value or structure: {return_value_from_process}")
                continue

            refined_image_np = None
            if isinstance(controlnet_outputs, list) and len(controlnet_outputs) >= 1:
                refined_image_np = controlnet_outputs[0]
            else:
                print(f"WARN: controlnet_outputs is not a non-empty list: {controlnet_outputs}")
                continue

            if refined_image_np is None:
                 print(f"WARN: refined_image_np is None *after extraction* for view {idx}. Skipping.")
                 continue


            if refined_image_np is not None and isinstance(refined_image_np, np.ndarray):
                refined_image_pil = Image.fromarray(refined_image_np)
                resized_image_pil = refined_image_pil.resize((img_W, img_H), Image.Resampling.LANCZOS)
                refined_image_tensor = to_tensor(resized_image_pil).cuda()
                controlnet_targets.append(refined_image_tensor)
            else:
                print(f"WARN: process returned invalid or None output for view {idx}. Skipping.")
                continue

        except AttributeError as ae:
            print(f"--- CAUGHT AttributeError ---")
            print(f"ERROR during Image.fromarray for view {idx}: {ae}")
            continue
        except Exception as e_outer_loop:
            print(f"--- CAUGHT Exception in outer loop for view {idx} ---")
            print(f"ERROR: {e_outer_loop}")
            import traceback
            traceback.print_exc()
            continue

        if idx < 5: # Save first few
            debug_save_path = os.path.join(args.model_path, "debug_stage2_gen")
            os.makedirs(debug_save_path, exist_ok=True)
            torchvision.utils.save_image(rendered_image_gs, os.path.join(debug_save_path, f"virtual_{idx}_gs_render.png"))
            torchvision.utils.save_image(refined_image_tensor, os.path.join(debug_save_path, f"virtual_{idx}_controlnet_target.png"))

    if not controlnet_targets:
            print("[ERROR] Failed to generate any ControlNet targets.")
            sys.exit(1)
    print(f"Generated {len(controlnet_targets)} ControlNet target images.")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_lpips_loss_for_log = 0.0

    total_iterations = opt.iterations
    num_refinement_iterations = opt.iterations
    end_iteration = first_iter + num_refinement_iterations
    print(f"[INFO] Starting refinement from iter {first_iter}, running for {num_refinement_iterations} steps, ending at iter {end_iteration}.")

    progress_bar = tqdm(range(first_iter, end_iteration + 1), desc="Stage 2 Refinement")

    for iteration in range(first_iter, end_iteration + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0: gaussians.oneupSHdegree()

        # Reconstruction Loss REAL VIEWS
        if not viewpoint_stack: viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg_real = render(viewpoint_cam, gaussians, pipe, background)
        image_real = render_pkg_real["render"]
        gt_image_real = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image_real, gt_image_real)
        loss_ssim = ssim(image_real.unsqueeze(0), gt_image_real.unsqueeze(0))
        loss_recon = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - loss_ssim)

        # Novel View Supervision Loss VIRTUAL VIEWS
        virtual_idx = randint(0, len(virtual_Rs) - 1)

        try:
             virtual_cam = Camera(colmap_id=virtual_idx, R=virtual_Rs[virtual_idx], T=virtual_Ts[virtual_idx],
                                 FoVx=sample_cam.FoVx, FoVy=sample_cam.FoVy,
                                 image=torch.zeros(3, img_H, img_W), gt_alpha_mask=None,
                                 image_name=f"virtual_{virtual_idx}", uid=-virtual_idx-1,
                                 data_device="cuda",
                                 mono_depth=None )
        except Exception as e:
             print(f"\n[ERROR] Failed to create virtual Camera object in loop: {e}")
             continue

        render_pkg_virtual = render(virtual_cam, gaussians, pipe, background)
        rendered_image_virtual = render_pkg_virtual["render"].clamp(0.0, 1.0)

        target_image_controlnet = controlnet_targets[virtual_idx]

        loss_lpips = loss_fn_lpips(rendered_image_virtual.unsqueeze(0), target_image_controlnet.unsqueeze(0)).mean()

        total_loss = loss_recon + args.lambda_lpips * loss_lpips

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            ema_lpips_loss_for_log = 0.4 * loss_lpips.item() + 0.6 * ema_lpips_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "LPIPS": f"{ema_lpips_loss_for_log:.4f}",
                    "n": len(gaussians._xyz)
                })
                progress_bar.update(10)
            if iteration == total_iterations:
                progress_bar.close()

            if tb_writer:
                    tb_writer.add_scalar('stage2_loss/total_loss', total_loss.item(), iteration)
                    tb_writer.add_scalar('stage2_loss/recon_loss', loss_recon.item(), iteration)
                    tb_writer.add_scalar('stage2_loss/lpips_loss', loss_lpips.item(), iteration)
                    tb_writer.add_scalar('stage2/iter_time', iter_start.elapsed_time(iter_end), iteration)
                    tb_writer.add_scalar('stage2/total_points', len(gaussians._xyz), iteration)

            if (iteration in save_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians (Stage 2)")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and len(gaussians._xyz) < opt.max_num_splats:
                visibility_filter = render_pkg_real["visibility_filter"] # Use visibility from real view
                radii = render_pkg_real["radii"]
                viewspace_point_tensor = render_pkg_real["viewspace_points"]

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # Include outlier removal if used in traings.py
                if opt.remove_outliers_interval > 0 and iteration % opt.remove_outliers_interval == 0:
                     gaussians.remove_outliers(opt, iteration, linear=True)


            # Optimizer step
            if iteration < end_iteration:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Checkpointing
            if (iteration in args.checkpoint_iterations): # Check if this is intended behaviour
                print(f"\n[ITER {iteration}] Saving Checkpoint (Stage 2)")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt_stage2_" + str(iteration) + ".pth")

    print("Stage 2 Refinement finished.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 2 Refinement Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None, required=True, help="Path to the GS model checkpoint from Stage 1 (e.g., output/.../ckpt30000.pth)")
    parser.add_argument('--controlnet_model_name', type=str, default='control_v11f1e_sd15_tile', help="ControlNet model variant name")
    parser.add_argument('--sd_v15_path', type=str, default='./models/v1-5-pruned.ckpt', help="Path to base Stable Diffusion v1.5 checkpoint")
    parser.add_argument('--lora_checkpoint_path', type=str, required=True, help="Path to the trained LoRA checkpoint")
    parser.add_argument('--lora_rank', type=int, default=64, help="Rank for LoRA matrices (must match training)")
    parser.add_argument('--prompt', type=str, default='high quality, sharp scene', help="Text prompt for ControlNet guidance")


    parser.add_argument('--num_virtual_views', type=int, default=10, help="Number of virtual views to generate")
    parser.add_argument('--lambda_lpips', type=float, default=0.1, help="Weight for the LPIPS novel view supervision loss")
    parser.add_argument('--ddim_steps', type=int, default=50, help="DDIM steps for ControlNet inference")
    parser.add_argument('--control_strength', type=float, default=0.8, help="ControlNet strength")
    parser.add_argument('--guidance_scale', type=float, default=1.0, help="Guidance scale for ControlNet")
    parser.add_argument('--ddim_eta', type=float, default=1.0, help="DDIM eta parameter")
    parser.add_argument('--sparse_train_file', type=str, default=None,
                           help="Path to a text file listing specific image base names (one per line) from the original dataset. Only these cameras will be used as the basis for virtual view interpolation.")


    args = parser.parse_args(sys.argv[1:])
    # Ensure total iterations are defined
    if hasattr(op.extract(args), 'iterations'):
            args.save_iterations.append(op.extract(args).iterations)
            args.checkpoint_iterations.append(op.extract(args).iterations) # Save final checkpoint
    else:
            print("[WARN] Could not determine total iterations from OptimizationParams. Final save/checkpoint might not occur.")

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training_refine(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        test_iterations=args.test_iterations,
        save_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        start_checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        args=args
    )

    print("\nStage 2 Refinement Training complete.")