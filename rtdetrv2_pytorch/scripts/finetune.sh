export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --master_port=9909 --nproc_per_node=8 tools/train.py \
    -c ./configs/rtdetrv2/rtdetrv2_r101vd_6x_kitti.yml \
    -t ./ckpts/rtdetrv2_r101vd_6x_coco_from_paddle.pth \
    --use-amp --seed=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --master_port=9909 --nproc_per_node=8 tools/train.py \
#     -c ./configs/rtdetrv2/rtdetrv2_r101vd_6x_kitti.yml \
#     -t ./ckpts/rtdetrv2_r101vd_6x_coco_from_paddle.pth \
#     --use-amp --seed=0
