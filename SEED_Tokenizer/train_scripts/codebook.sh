export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1


OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 18272 --nnodes=8 --node_rank=$1 --master_addr=$2 --use-env \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_codebook.yaml

