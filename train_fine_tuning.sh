#/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train_fine_tuning.py \
    --resume=model/ckpt_e160.pth \
    #--resume=False \
