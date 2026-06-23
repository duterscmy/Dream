#!/bin/bash
#SBATCH --job-name=eval_gsm8k_base
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=a100

source ~/.bashrc
conda activate ttrl_env
cd /mnt/fast/nobackup/scratch4weeks/mc03002/prophet
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
mkdir -p logs
mkdir -p evals_results/baseline

length=256
block=32

model=/mnt/fast/nobackup/scratch4weeks/mc03002/models/Dream-v0-Base-7B-origin-block
cp generate_functions/generation_utils.origin_block.py $model/generation_utils.py
max_new_tokens=256
echo "====gsm8k standard block ${max_new_tokens}===="
accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=${model},trust_remote_code=True,max_new_tokens=${max_new_tokens},diffusion_steps=${max_new_tokens},dtype="bfloat16",temperature=0.0,alg="maskgit_plus" \
    --tasks gsm8k_cot_zeroshot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "evals_results/gsm8k-standard-len${max_new_tokens}_0shot" \
    --log_samples --confirm_run_unsafe_code &> "logs/gsm8k-standard-len${max_new_tokens}_0shot.log"