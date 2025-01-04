export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=0
export NCCL_IB_TIMEOUT=0
torchrun --master_port=9909 --nproc_per_node=8 tools/train.py \
    -c ./configs/rtdetrv2/rtdetrv2_r101vd_6x_kitti.yml \
    --use-amp --seed=0
