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
cd /mnt/fast/nobackup/scratch4weeks/mc03002/dream
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0

mkdir -p logs
mkdir -p evals_results/adaptive_block

length=256
block=32
num_fewshot=3
threshold=0.9

model=/mnt/fast/nobackup/scratch4weeks/mc03002/models/Dream-v0-Base-7B-adaptive-block
cp generate_functions/generation_utils.adaptive_block.py $model/generation_utils.py
max_new_tokens=256
echo "====gsm8k adaptive block ${max_new_tokens}===="
accelerate launch eval.py --model dream  \
    --model_args pretrained=${model},threshold=${threshold},print_all_token_records=false,add_bos_token=true,trust_remote_code=True,max_new_tokens=${max_new_tokens},diffusion_steps=${max_new_tokens},dtype="bfloat16",temperature=0.0,alg="maskgit_plus" \
    --tasks gsm8k_cot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot ${num_fewshot} \
    --output_path "evals_results/adaptive_block/gsm8k-adaptive-block-threshold${threshold}-len${max_new_tokens}_${num_fewshot}shot" \
    --log_samples --confirm_run_unsafe_code &> "logs/gsm8k-adaptive-block-threshold${threshold}-len${max_new_tokens}_${num_fewshot}shot.log"