#/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --resume=model/ckpt_e100.pth \
    --testResDir=result/tmp \
    #--resume=pretrained/indexnet_matting.pth-author.tar \
