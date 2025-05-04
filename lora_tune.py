import argparse
import os
import torch
import pytorch_lightning as pl

from functools import partial
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger, LoraCheckpoint

from gs_dataset import GSCachedDataset
from torch.utils.data import DataLoader
from minlora import add_lora, LoRAParametrization
import time

_ = torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ControlNet LoRA using GS hints (train_repairmodel style).')

    parser.add_argument('--model_name', type=str, default='control_v11f1e_sd15_tile', help="ControlNet model variant")
    parser.add_argument('--v15_path', type=str, default='./models/v1-5-pruned.ckpt', help="Path to base Stable Diffusion v1.5 checkpoint")
    parser.add_argument('--controlnet_path', type=str, default=None, help="Optional path to ControlNet weights (e.g., .pth file). If None, uses model_name.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing target images and COLMAP/camera poses")
    parser.add_argument('--hint_dir', type=str, required=True, help="Path to the base directory containing PRE-RENDERED hint images subdirs (e.g., output/model/intermediate_hints).")
    parser.add_argument('--image_size', type=int, default=512, help="Target image size for training")
    parser.add_argument('--prompt', type=str, default='high quality, sharp scene', help="Base text prompt")
    parser.add_argument('--use_prompt_list', action='store_true', default=False, help="Use predefined templates for prompts")
    parser.add_argument('--bg_white', action='store_true', default=False, help="Use white background for padding.")
    parser.add_argument('--cache_max_iter', type=int, default=50, help="Maximum number of hint images to cache per view.")

    # Training related
    parser.add_argument('--exp_name', type=str, default='output/controlnet_lora_finetune/experiment', help="Base directory for saving logs and checkpoints")
    parser.add_argument('--batch_size', type=int, default=1, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--lora_rank', type=int, default=64, help="Rank for LoRA matrices")
    parser.add_argument('--max_steps', type=int, default=1800, help="Total training steps")
    parser.add_argument('--callbacks_every_n_train_steps', type=int, default=600, help="Frequency for logging images and saving checkpoints")
    parser.add_argument('--precision', type=str, default='32', choices=['bf16-mixed', '16-mixed', '32'], help="Training precision")
    parser.add_argument('--sparse_train_file', type=str, default=None,
                        help="Optional path to a text file containing image filenames (one per line) to use for training.")

    args = parser.parse_args()


    model_config_path = f'./models/{args.model_name}.yaml'
    controlnet_weights_path = args.controlnet_path if args.controlnet_path else f'./models/{args.model_name}.pth'
    model = create_model(model_config_path).cpu()
    model.load_state_dict(load_state_dict(args.v15_path, location='cpu'), strict=False)
    model.load_state_dict(load_state_dict(controlnet_weights_path, location='cpu'), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = True
    model.only_mid_control = False 
    model.train_lora = True

    # Freeze Original Weights
    print("Freezing original model weights...")
    for param in model.model.diffusion_model.parameters(): param.requires_grad = False
    for param in model.control_model.parameters(): param.requires_grad = False
    for param in model.cond_stage_model.parameters(): param.requires_grad = False
    print("Original weights frozen.")

    # LoRA Application
    print(f"Applying LoRA with rank {args.lora_rank}...")
    lora_config = {
        nn.Embedding: { "weight": partial(LoRAParametrization.from_embedding, rank=args.lora_rank)},
        nn.Linear: { "weight": partial(LoRAParametrization.from_linear, rank=args.lora_rank)},
        nn.Conv2d: { "weight": partial(LoRAParametrization.from_conv2d, rank=args.lora_rank)}
    }
    for name, module in model.model.diffusion_model.named_modules():
        if name.endswith('transformer_blocks'): add_lora(module, lora_config=lora_config)
    for name, module in model.control_model.named_modules():
        if name.endswith('transformer_blocks'): add_lora(module, lora_config=lora_config)
    add_lora(model.cond_stage_model, lora_config=lora_config)


    start_time = time.time()
    exp_path = args.exp_name
    os.makedirs(exp_path, exist_ok=True)

    print("Initializing Dataset using hint pool (train_repairmodel style)...")
    dataset = GSCachedDataset(        
        data_dir=args.data_dir,
        hint_dir=args.hint_dir,
        image_size=args.image_size,
        prompt=args.prompt,
        bg_white=args.bg_white,
        use_prompt_list=args.use_prompt_list,
        sparse_train_file=args.sparse_train_file
    )
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset initialized. Number of views/targets: {len(dataset)}")
    data_init_time = time.time()
    print(f"Dataset initialization time: {data_init_time - start_time:.2f}s")

    # Logging -
    loggers = [ TensorBoardLogger(os.path.join(exp_path, 'tf_logs')) ]
    callbacks = [
        ImageLogger(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps,
                    log_images_kwargs = {"plot_diffusion_rows": True, "sample": True}),
        LoraCheckpoint(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps)
    ]

    # Trainer Setup
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=args.precision,
        logger=loggers,
        callbacks=callbacks,
        max_steps=args.max_steps,
        check_val_every_n_epoch=None,
        val_check_interval = args.callbacks_every_n_train_steps
    )

    # Verify Trainable Parameters (bug fixing)
    print("\nVerifying trainable parameters after LoRA application:")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
          total_params += param.numel()
          if param.requires_grad:
                print(f"  Trainable: {name} ({param.numel()})")
                trainable_params += param.numel()
    print(f"\nTotal model parameters: {total_params}")
    print(f"Total trainable parameters (LoRA): {trainable_params}")
    if total_params > 0: print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")

    # Training 
    training_start_time = time.time()
    trainer.fit(model, dataloader)
    training_end_time = time.time()

    end_time = time.time()
    print(f"\nTraining finished.")
    print(f"Dataset Init Time: {data_init_time - start_time:.4f} seconds.")
    print(f"Training Time: {training_end_time - training_start_time:.4f} seconds.")
    print(f"Total Script Time: {end_time - start_time:.4f} seconds.")