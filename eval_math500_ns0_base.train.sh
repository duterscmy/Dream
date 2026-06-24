#!/bin/bash
#SBATCH --job-name=eval_math500_base
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=a100

source ~/.bashrc
conda activate ttrl_env
cd /mnt/fast/nobackup/scratch4weeks/mc03002/dream
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
mkdir -p logs
mkdir -p evals_results/standard_block

length=256
block=32
num_fewshot=3

model=/mnt/fast/nobackup/scratch4weeks/mc03002/models/Dream-v0-Base-7B-origin-block
cp generate_functions/generation_utils.origin_block.py $model/generation_utils.py
max_new_tokens=256

echo "====math500 standard block ${max_new_tokens}===="
accelerate launch eval.py --model dream  \
    --model_args pretrained=${model},add_bos_token=true,trust_remote_code=True,max_new_tokens=${max_new_tokens},diffusion_steps=${max_new_tokens},dtype="bfloat16",temperature=0.0,alg="maskgit_plus" \
    --tasks minerva_math500_train \
    --limit 5000 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot ${num_fewshot} \
    --output_path "evals_results/standard_block/math500-train-standard-block-len${max_new_tokens}_${num_fewshot}shot" \
    --log_samples --confirm_run_unsafe_code &> "logs/math500-train-standard-block-len${max_new_tokens}_${num_fewshot}shot.log"