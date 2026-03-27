#!/bin/bash

model="/home/work/mt_cmy/diffullama/LLaMA-Factory/dream/models/Dream-v0-Base-7B-Adaptive-Parallel-Beam-Search"

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

    echo "====gpqa adaptive_parallel_beam2 ${l}===="
    accelerate launch --main_process_port ${current_port} eval.py --model dream \
        --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},add_bos_token=true,temperature=0.0,top_p=0.95,torch_dtype=torch.bfloat16" \
        --tasks gpqa_main_cot_n_shot \
        --num_fewshot 4 \
        --batch_size 1 \
        --output_path "evals_results/gpqa-len${l}_ns4_adaptive_parallel_beam2" \
        --log_samples \
        --confirm_run_unsafe_code &> "logs/gpqa-len${l}_ns4_adaptive_parallel_beam2.log"

    # echo "====math adaptive_parallel_beam2 ${l}===="
    # accelerate launch --main_process_port ${current_port} eval.py --model dream \
    #     --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},add_bos_token=true,temperature=0.0,top_p=0.95,torch_dtype=torch.bfloat16" \
    #     --tasks hendrycks_math500 \
    #     --num_fewshot 4 \
    #     --batch_size 1 \
    #     --output_path "evals_results/math-len${l}_ns3_adaptive_parallel_beam2" \
    #     --log_samples \
    #     --confirm_run_unsafe_code &> "logs/math-len${l}_ns3_adaptive_parallel_beam2.log"
    
    # echo "====humaneval adaptive parallel beam search ${l}===="
    # accelerate launch --main_process_port ${current_port} eval.py --model dream \
    #     --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true" \
    #     --tasks humaneval \
    #     --num_fewshot 0 \
    #     --batch_size 1 \
    #     --output_path "evals_results/humaneval-len${l}_adaptive_parallel_beam2" \
    #     --log_samples \
    #     --confirm_run_unsafe_code &> "logs/humaneval-len${l}_adaptive_parallel_beam2.log"
    
    # # 递增端口号
    # current_port=$((port))
    # port=$((port + 1))
    
    # echo "====mbpp adaptive parallel beam search ${l}===="
    # accelerate launch --main_process_port ${current_port} eval.py --model dream \
    #     --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},temperature=0.0,top_p=0.95,add_bos_token=true" \
    #     --tasks mbpp \
    #     --num_fewshot 3 \
    #     --batch_size 1 \
    #     --output_path "evals_results/mbpp-len${l}_ns3_adaptive_parallel_beam2" \
    #     --log_samples \
    #     --confirm_run_unsafe_code &> "logs/mbpp-len${l}_ns3_adaptive_parallel_beam2.log"
    
    # # 递增端口号
    # current_port=$((port))
    # port=$((port + 1))
    
    # echo "====gsm8k adaptive parallel beam search ${l}===="
    # accelerate launch --main_process_port ${current_port} eval.py --model dream \
    #     --model_args "pretrained=${model},max_new_tokens=${l},diffusion_steps=${diffusion_steps},add_bos_token=true,temperature=0.0,top_p=0.95,torch_dtype=torch.bfloat16" \
    #     --tasks gsm8k_cot \
    #     --num_fewshot 4 \
    #     --batch_size 1 \
    #     --output_path "evals_results/gsm8k-len${l}_ns4_adaptive_parallel_beam2" \
    #     --log_samples \
    #     --confirm_run_unsafe_code &> "logs/gsm8k-len${l}_ns4_adaptive_parallel_beam2.log"
    
    echo "完成长度 ${l} 的所有任务评估"
    echo "----------------------------------------"
done

echo "所有长度评估任务已完成！"

## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
