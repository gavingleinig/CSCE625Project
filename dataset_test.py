import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import traceback
from gs_dataset import GSCachedDataset

def denormalize_target(img_np):
    return np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)

def main(args):
    print("Attempting to initialize GSCachedDataset...")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  Hint Dir: {args.hint_dir}")
    print(f"  Image Size: {args.image_size}")
    print(f"  Sparse File: {args.sparse_train_file}")


    dataset = GSCachedDataset(
        data_dir=args.data_dir,
        hint_dir=args.hint_dir,
        image_size=args.image_size,
        prompt=args.prompt,
        bg_white=False,
        use_prompt_list=False,
        sparse_train_file=args.sparse_train_file
    )

    if len(dataset) == 0:
        print("\nERROR: Dataset initialized but contains 0 items.")
        return

    print(f"\nDataset loaded successfully with {len(dataset)} views.")

    num_samples_to_show = min(args.num_samples, len(dataset))
    if num_samples_to_show <= 0:
        print("Number of samples to show must be positive.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {num_samples_to_show} random samples to: {args.output_dir}")

    indices_to_show = random.sample(range(len(dataset)), num_samples_to_show)

    for i, idx in enumerate(indices_to_show):
        print(f"\n--- Processing sample {i+1}/{num_samples_to_show} (Dataset Index: {idx}) ---")
        try:
            sample = dataset[idx]
            hint_img = sample.get('hint')
            target_img = sample.get('jpg')
            prompt_txt = sample.get('txt', 'N/A')

            if hint_img is None or target_img is None:
                 print(f"WARNING: Missing 'hint' or 'jpg' key for index {idx}. Skipping.")
                 continue

            target_img_display = denormalize_target(target_img)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Sample {i+1}/{num_samples_to_show} (Dataset Index: {idx})\nPrompt: '{prompt_txt}'", fontsize=14)

            axes[0].imshow(hint_img)
            axes[0].set_title("Hint Image")
            axes[0].axis('off')

            axes[1].imshow(target_img_display)
            axes[1].set_title("Target Image")
            axes[1].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            output_filename = os.path.join(args.output_dir, f"sample_idx_{idx}.png")
            plt.savefig(output_filename)
            print(f"  Saved plot to: {output_filename}")

            plt.close(fig)

        except Exception as e:
            print(f"ERROR: Failed to process or save sample at index {idx}: {e}")
            traceback.print_exc()

    print("\nFinished saving samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GSCachedDataset: Load data and SAVE sample hint/target pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True, help="Path to COLMAP/images data.")
    parser.add_argument('--hint_dir', type=str, required=True, help="Path to base hint directory.")
    parser.add_argument('--image_size', type=int, default=512, help="Target image size.")
    parser.add_argument('--prompt', type=str, default='', help="Base text prompt.")
    parser.add_argument('--sparse_train_file', type=str, default=None, help="Optional sparse training file.")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of random samples to save.")

    parser.add_argument('--output_dir', type=str, default="./dataset_samples",
                        help="Directory to save the output sample images.")

    args = parser.parse_args()


    main(args)