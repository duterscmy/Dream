#!/bin/bash
#SBATCH --job-name="Dream_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # 请求2块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar


model="/lus/lfs1aip2/projects/public/u6er/mingyu/models/dream-exp-greedy"

export HF_ALLOW_CODE_EVAL=1
lengths=(256)  # 定义多个长度值

port=29510  # 起始端口号

for l in "${lengths[@]}"
do
    # 递增端口号避免冲突
    current_port=$((port))
    port=$((port + 1))
    
    # 计算 diffusion_steps
    diffusion_steps=$((l))

    
    echo "====humaneval greedy ${l}===="
    HF_ALLOW_CODE_EVAL=1 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=${model},trust_remote_code=True,max_new_tokens=${l},diffusion_steps=${diffusion_steps},dtype="bfloat16",temperature=0.1,top_p=0.9,alg="maskgit_plus" \
    --tasks humaneval_instruct \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "evals_results/humaneval-len${l}_greedy_tmp0.1" \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template #&> "logs/humaneval-len${l}_greedy.log"
    

    # echo "====mbpp greedy ${l}===="
    # HF_ALLOW_CODE_EVAL=1 accelerate launch --main_process_port 12334 -m lm_eval \
    # --model diffllm \
    # --model_args pretrained=${model},trust_remote_code=True,max_new_tokens=${l},diffusion_steps=${diffusion_steps},dtype="bfloat16",temperature=0.1,top_p=0.9,alg="maskgit_plus" \
    # --tasks mbpp_instruct \
    # --device cuda \
    # --batch_size 1 \
    # --num_fewshot 0 \
    # --output_path "evals_results/mbpp-len${l}_ns0_greedy" \
    # --log_samples --confirm_run_unsafe_code \
    # --apply_chat_template &> "logs/mbpp-len${l}_ns0_greedy.log"

    # # 递增端口号
    # current_port=$((port))
    # port=$((port + 1))
    
    # echo "====gsm8k greedy ${l}===="
    # accelerate launch --main_process_port 12334 -m lm_eval \
    # --model diffllm \
    # --model_args pretrained=${model},trust_remote_code=True,max_new_tokens=${l},diffusion_steps=${diffusion_steps},dtype="bfloat16",temperature=0.1,top_p=0.9,alg="maskgit_plus" \
    # --tasks gsm8k_cot \
    # --device cuda \
    # --batch_size 1 \
    # --num_fewshot 0 \
    # --output_path "evals_results/gsm8k-len${l}_ns4_greedy" \
    # --log_samples --confirm_run_unsafe_code \
    # --apply_chat_template &> "logs/gsm8k-len${l}_ns4_greedy.log"
    
    echo "完成长度 ${l} 的所有任务评估"
    echo "----------------------------------------"
done

echo "所有长度评估任务已完成！"

## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
