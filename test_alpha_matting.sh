#/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test_alpha_matting.py \
    --resume=model/ckpt_e100.pth \
    #--resume=pretrained/indexnet_matting.pth-author.tar \
