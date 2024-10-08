import argparse
from PIL import Image
import torch
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
import multiprocessing as mp

processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")


class ImagePromptDataset(Dataset):
    def __init__(self, image_paths, prompts):
        self.image_paths = image_paths
        self.prompts = prompts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        prompt = self.prompts[idx]
        return image, prompt


def batch_get_clip_scores(model, processor, images, prompts, device):
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

    diagonal_scores = logits_per_image.diag().tolist()
    return diagonal_scores


def read_prompt_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt


def list_images_and_prompts(image_dir):
    images = []
    prompts = []
    for file_name in tqdm(os.listdir(image_dir)):
        if file_name.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
            image_path = os.path.join(image_dir, file_name)
            txt_path = os.path.join(image_dir, os.path.splitext(file_name)[0] + ".txt")
            if os.path.isfile(txt_path):
                images.append(image_path)
                prompts.append(read_prompt_from_txt(txt_path))
    return images, prompts


def split_dataset_fixed(dataset, num_splits):
    """Split the dataset into fixed parts for each GPU"""
    length = len(dataset)
    split_size = length // num_splits
    subsets = []
    for i in range(num_splits):
        start_idx = i * split_size
        # Ensure the last split takes the remaining samples
        end_idx = length if i == num_splits - 1 else (i + 1) * split_size
        subsets.append(Subset(dataset, range(start_idx, end_idx)))
    return subsets


def custom_collate_fn(batch):
    """Custom collate function to handle PIL Image and text batch"""
    images, prompts = zip(*batch)  # Unzip the batch
    return list(images), list(prompts)


def worker(rank, dataset, batch_size, device_id, return_dict):
    # Set device for the process
    device = torch.device(f"cuda:{device_id}")

    # Initialize model
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").to(
        device
    )

    # Create a DataLoader for batch processing
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    scores = []

    # Process each batch of data
    try:
        for images, batch_prompts in tqdm(dataloader, total=len(dataloader)):
            batch_scores = batch_get_clip_scores(
                model, processor, images, batch_prompts, device
            )
            scores.extend(batch_scores)
        return_dict[rank] = scores
    except Exception as e:
        print(f"Error processing on device {device_id}: {e}")
        return_dict[rank] = []  # Ensure a default empty list if an error occurs


def main(image_dir, batch_size, num_gpus):
    # Load all images and prompts
    img_paths, prompts = list_images_and_prompts(image_dir)

    if not img_paths:
        raise ValueError("No images found in the specified directory.")

    # Create the full dataset
    full_dataset = ImagePromptDataset(img_paths, prompts)

    # Split dataset into fixed subsets for each GPU
    dataset_splits = split_dataset_fixed(full_dataset, num_gpus)
    for subset in dataset_splits:
        print(len(subset))

    # Create a manager for shared memory dictionary
    manager = mp.Manager()
    return_dict = manager.dict()

    # Create processes for each GPU
    processes = []
    for rank in range(num_gpus):
        subset = dataset_splits[rank]
        p = mp.Process(
            target=worker, args=(rank, subset, batch_size, rank, return_dict)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Combine results from all processes
    all_scores = []
    for rank in range(num_gpus):
        if rank in return_dict:
            all_scores.extend(return_dict[rank])
        else:
            print(f"Warning: No data returned from rank {rank}")

    mean_score = np.mean(all_scores) if all_scores else float("nan")
    print(f"{image_dir}")
    print(f"Mean CLIP Score: {mean_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean CLIP score for images and prompts."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use.",
    )

    args = parser.parse_args()
    image_dirs = [
        "path_1",
        "path_2",
        "path_3",
        "path_...",
    ]
    for image_dir in image_dirs:
        try:
            main(image_dir, args.batch_size, args.num_gpus)
        except:
            print(f"failed at {image_dir}")
