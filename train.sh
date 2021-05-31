#/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
    --resume=False \
    #--resume=model/ckpt_e10.pth \
