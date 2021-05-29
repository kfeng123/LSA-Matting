#/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
    --batchSize=16 \
    --nEpochs=200 \
    --lr=5e-5 \
    --pretrain=False \
    --resume=False \
    #--resume=model/ckpt_e26.pth \
    #--pretrain=pretrained/indexnet_matting.pth-author.tar \
