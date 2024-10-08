PREFIX=rectified_diffusion_phased_xl
MODEL_DIR="/mnt/wangfuyun/weights/stable-diffusion-xl-base-1.0"
OUTPUT_DIR="outputs_formal/$PREFIX"
PROJ_NAME="$PREFIX"
accelerate launch --main_process_port 29506  train_split_sdxl.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --pretrained_vae_model_name_or_path="/mnt/wangfuyun/weights/sdxl-vae-fp16-fix" \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --learning_rate=5e-6 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=20000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --validation_steps=1000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=20 \
    --train_batch_size=5 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --max_grad_norm=1 \
    --gradient_checkpointing 
