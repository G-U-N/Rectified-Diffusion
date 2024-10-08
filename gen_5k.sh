python generate_5k.py --generation_path="results/rectifieddiffusion/5k/rd_cfg1.5_1step" --personalized_path="weights/rd.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 1

python generate_5k.py --generation_path="results/rectifieddiffusion/5k/rd_cfg1.5_2step" --personalized_path="weights/rd.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 2

python generate_5k.py --generation_path="results/rectifieddiffusion/5k/rd_cfg1.5_4step" --personalized_path="weights/rd.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 4

python generate_5k.py --generation_path="results/rectifieddiffusion/5k/rd_cfg1.5_8step" --personalized_path="weights/rd.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 8

python generate_5k.py --generation_path="results/rectifieddiffusion/5k/rd_cfg1.5_16step" --personalized_path="weights/rd.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 16


python generate_5k.py --generation_path="results/rectifieddiffusion/5k/rd_cfg1.5_25step" --personalized_path="weights/rd.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 25



python generate_5k.py --generation_path="results/cm/5k/cfg1.0_1step" --personalized_path="weights/cm.ckpt" \
    --cfg=1.0 --seed 0 --num_inference_steps 1

python generate_5k.py --generation_path="results/cm/5k/cfg1.0_2step" --personalized_path="weights/cm.ckpt" \
    --cfg=1.0 --seed 0 --num_inference_steps 2


python generate_5k.py --generation_path="results/phased/5k/cfg1.5_4step" --personalized_path="weights/phased.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 4

python generate_5k.py --generation_path="results/phasedxl/5k/cfg1.5_4step" --personalized_path="weights/phasedxl.ckpt" \
    --cfg=1.5 --seed 0 --num_inference_steps 4 --resolution=1024

