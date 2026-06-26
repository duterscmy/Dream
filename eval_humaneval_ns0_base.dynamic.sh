#!/bin/bash
#SBATCH --job-name=eval_humaneval_dynamic
#SBATCH --time=1:00:00
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
export HF_ALLOW_CODE_EVAL=1

mkdir -p logs
mkdir -p evals_results/dynamic_block

length=256
block=32
num_fewshot=0
threshold=0.9
correct_ratio=100.0
calibrated_threshold=token_threshold_on_trainset/global_dream_3shot_token_threshold_grid_p${correct_ratio}_mincount200_minaccepted100.json

model=/mnt/fast/nobackup/scratch4weeks/mc03002/models/Dream-v0-Base-7B-dynamic-block
cp generate_functions/generation_utils.dynamic_block.py $model/generation_utils.py
max_new_tokens=256

echo "====humaneval dynamic block ${max_new_tokens}===="
accelerate launch eval.py --model dream  \
    --model_args pretrained=${model},threshold=${threshold},calibrated_threshold=${calibrated_threshold},print_all_token_records=false,escape_until=true,add_bos_token=true,trust_remote_code=True,max_new_tokens=${max_new_tokens},diffusion_steps=${max_new_tokens},dtype="bfloat16",temperature=0.0,alg="maskgit_plus" \
    --tasks humaneval \
    --device cuda \
    --batch_size 1 \
    --num_fewshot ${num_fewshot} \
    --output_path "evals_results/dynamic_block/humaneval-dynamic-block-global-threshold${threshold}-correct_ratio${correct_ratio}-len${max_new_tokens}_${num_fewshot}shot" \
    --log_samples --confirm_run_unsafe_code &> "logs/humaneval-dynamic-block-global-threshold${threshold}-correct_ratio${correct_ratio}-len${max_new_tokens}_${num_fewshot}shot.log"