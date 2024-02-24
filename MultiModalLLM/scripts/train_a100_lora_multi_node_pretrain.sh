
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=info

torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=20002 --node_rank=$INDEX src/train/train.py \
    --model configs/model/vicuna_7b_lora.yaml \
    --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml \
    --train_data configs/data/multi_torchdata_pretrain.yaml \
    --output_dir log/seed_vicuna-7b_lora_pretrain \
    --deepspeed configs/deepspeed/stage2_bf16.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_steps 30000 \
    --min_lr_ratio 0.05 \
    --learning_rate 1.5e-4 \
    --weight_decay 5e-2 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --report_to "tensorboard" \
    --gradient_checkpointing \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --log_level 'info' \
    --logging_nan_inf_filter "no" \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip \