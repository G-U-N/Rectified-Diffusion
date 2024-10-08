from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import argparse
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch
from safetensors.torch import load_file
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, EulerDiscreteScheduler
import time
import numpy as np
import json

num_device = 8
num_processes = 8

from diffusers import StableDiffusionXLPipeline


import torch, torchvision
from diffusers import UNet2DConditionModel

# from perflow.src.utils_perflow import merge_delta_weights_into_unet
# from perflow.src.scheduler_perflow import PeRFlowScheduler

# from InstaFlow.code.pipeline_rf import RectifiedFlowPipeline


def extract_image_caption_pairs(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    image_captions = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]

        if image_id not in image_captions:
            image_captions[image_id] = []

        image_captions[image_id].append(caption)

    image_files = {}
    for image in data["images"]:
        image_id = image["id"]
        file_name = image["file_name"]
        image_files[image_id] = file_name

    img_paths = []
    captions = []
    for image_id, caption_list in image_captions.items():
        if image_id in image_files:
            file_name = image_files[image_id]
            for caption in caption_list[:1]:
                img_paths.append(file_name)
                captions.append(caption)

    return img_paths, captions


def read_prompts(file_path):
    json_file_path = "../coco-2017/annotations/captions_val2017.json"
    img_paths, captions = extract_image_caption_pairs(json_file_path)
    return captions, img_paths


def get_module_kohya_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)

    return kohya_ss_state_dict


def load_perflow_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    delta_weights = UNet2DConditionModel.from_pretrained(
        "perflow/perflow-sd15-delta-weights",
        torch_dtype=torch.float16,
        variant="v0-0",
    ).state_dict()
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_path, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = merge_delta_weights_into_unet(pipe, delta_weights)
    pipe.scheduler = PeRFlowScheduler.from_config(
        pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4
    )
    pipe.to(device, torch.float16)
    pipe.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    return pipe


def load_instaflow_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    pretrained_path = "InstaFlow/instaflow_0_9B_from_sd_1_5"
    pipe = RectifiedFlowPipeline.from_pretrained(
        pretrained_path, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to(device, torch.float16)
    pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    return pipe


def load_rectifiedflow_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    pretrained_path = "InstaFlow/2_rectified_flow_from_sd_1_5"
    pipe = RectifiedFlowPipeline.from_pretrained(
        pretrained_path, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to(device, torch.float16)
    pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    return pipe


def load_perflowxl_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "perflow/perflow-sdxl-base",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="v0-fix",
    )

    vae_weight = load_file(
        "/mnt/wangfuyun/weights/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors",
        "cpu",
    )
    pipe.vae.load_state_dict(vae_weight)
    pipe.scheduler = PeRFlowScheduler.from_config(
        pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4
    )
    pipe.to(device, torch.float16)
    pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    return pipe


def load_sdxl_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_path, torch_dtype=torch.float16
    )

    vae_weight = load_file(
        "/mnt/wangfuyun/weights/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors",
        "cpu",
    )

    pipe.vae.load_state_dict(vae_weight)
    del vae_weight
    pipe.scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        timestep_spacing="trailing",
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe.to(device, dtype=torch.float16)

    pipe.set_progress_bar_config(disable=True)

    if personalized_path:
        weight = torch.load(personalized_path, map_location="cpu")
        pipe.unet.load_state_dict(weight)
        del weight

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    return pipe


def load_pipeline_phased(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_path,
        scheduler=DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            timestep_spacing="trailing",
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        ),
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )

    pipeline.set_progress_bar_config(disable=True)
    if personalized_path:
        weight = torch.load(personalized_path, map_location="cpu")
        pipeline.unet.load_state_dict(weight)
        del weight

    pipeline = pipeline.to(device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_vae_slicing()

    return pipeline


def load_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device
):

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_path,
        scheduler=DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            timestep_spacing="trailing",
            clip_sample=False,
            set_alpha_to_one=False,
            beta_schedule="linear",
        ),
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )

    pipeline.set_progress_bar_config(disable=True)
    if personalized_path:
        weight = torch.load(personalized_path, map_location="cpu")
        pipeline.unet.load_state_dict(weight)
        del weight

    pipeline = pipeline.to(device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_vae_slicing()

    return pipeline


from multiprocessing import Pool


def process_image(args):
    img_path, transform, resolution, validation_path = args
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    path = os.path.join(validation_path, f"{img_path.split('/')[-1].split('.')[0]}.png")
    img.save(path)


def prepare_validation_set(validation_path, img_paths, resolution):
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    print("## Prepare validation dataset")
    transform = transforms.Compose(
        [
            transforms.Resize(
                resolution[0], interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(resolution),
        ]
    )

    args_list = [
        (img_path, transform, resolution, validation_path) for img_path in img_paths
    ]

    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))


def generate_batch_images(
    prompts,
    batch_size,
    resolution,
    pipeline,
    cfg,
    num_inference_steps,
    eta,
    device,
    device_id,
    weight_dtype,
    seed,
    generation_path,
):

    total_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )
    for batch_idx in tqdm(range(total_batches)):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        generator = torch.Generator(device=device).manual_seed(
            seed + batch_idx
        )  # Ensure different seeds for different batches

        with torch.autocast("cuda", weight_dtype):
            outputs = pipeline(
                prompt=batch_prompts,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=cfg,
                height=resolution[0],
                width=resolution[1],
            )
            images = outputs.images
        for img_idx, (img, prompt) in enumerate(zip(images, batch_prompts)):
            img_path = os.path.join(
                generation_path,
                f"{device_id}_{batch_idx * batch_size + img_idx:08d}.png",
            )
            img.save(img_path)
            text_path = os.path.join(
                generation_path,
                f"{device_id}_{batch_idx * batch_size + img_idx:08d}.txt",
            )
            with open(text_path, "w") as f:
                f.write(prompt)


def generate_imgs(
    generation_path,
    prompts,
    resolution,
    pipeline,
    cfg,
    num_inference_steps,
    eta,
    device_id,
    weight_dtype,
    seed,
):

    torch.cuda.set_device(f"cuda:{device_id%num_device}")
    device = torch.device(f"cuda:{device_id%num_device}")

    num_prompts_per_device = len(prompts) // num_processes
    start_idx = device_id * num_prompts_per_device
    end_idx = (
        start_idx + num_prompts_per_device
        if device_id != (num_processes - 1)
        else len(prompts)
    )

    device_prompts = prompts[start_idx:end_idx]

    print(f"Device {device} generating for prompts {start_idx} to {end_idx-1}")

    print("## Prepare generation dataset")
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    batch_size = 24
    generate_batch_images(
        device_prompts,
        batch_size,
        resolution,
        pipeline,
        cfg,
        num_inference_steps,
        eta,
        device,
        device_id,
        weight_dtype,
        seed,
        generation_path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_path", default="coco_5k")
    parser.add_argument("--generation_path", default="train_coco")
    parser.add_argument(
        "--pretrained_path", default="/mnt/wangfuyun/weights/stable-diffusion-v1-5"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cfg", default=1, type=float)
    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--eta", default=1, type=float)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--personalized_path", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.generation_path, exist_ok=True)
    os.makedirs(args.validation_path, exist_ok=True)

    prompts, img_paths = read_prompts("coco-2017/annotations/captions_val2017.json")

    prompts = [prompt.strip() for prompt in prompts]
    img_paths = [os.path.join("coco-2017/val2017", img_path) for img_path in img_paths]
    prepare_validation_set(args.validation_path, img_paths, args.resolution)

    pipelines = []
    for i in range(num_processes):
        if "perflowxl" in args.generation_path:
            pipelines.append(
                load_perflowxl_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "perflow" in args.generation_path:
            pipelines.append(
                load_perflow_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "phasedxl" in args.generation_path:
            pipelines.append(
                load_sdxl_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "instaflow" in args.generation_path:
            pipelines.append(
                load_instaflow_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "rectifiedflow" in args.generation_path:
            pipelines.append(
                load_rectifiedflow_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "rectifieddiffusion" in args.generation_path:
            pipelines.append(
                load_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "cm" in args.generation_path:
            pipelines.append(
                load_pipeline(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )
        elif "phased" in args.generation_path:
            pipelines.append(
                load_pipeline_phased(
                    args.pretrained_path,
                    None,
                    args.personalized_path,
                    torch.float16,
                    f"cuda:{i%num_device}",
                )
            )

        else:
            assert 0, "not implemented"

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                generate_imgs,
                args.generation_path,
                prompts,
                args.resolution,
                pipelines[device_id],
                args.cfg,
                args.num_inference_steps,
                args.eta,
                device_id,
                torch.float16,
                args.seed,
            )
            for device_id in range(num_processes)
        ]

        for future in as_completed(futures):
            print(f"Task completed: {future.result()}")
