# IMPROVING DEEP IMAGE MATTING VIA LOCAL SMOOTHNESS ASSUMPTION
This is the official repository of [**IMPROVING DEEP IMAGE MATTING VIA LOCAL SMOOTHNESS ASSUMPTION**](https://arxiv.org/abs/2112.13809).

This repo includes all source codes (including data preprocessing code, training code and testing code). 
Have fun!

To obtain the reestimated foreground, just run `python tools/reestimate_foreground_final.py`.

You can train the model on the [**Adobe Image Matting**](https://sites.google.com/view/deepimagematting) data using our training code.
To train the model,
first click [**here**](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth) 
to download the pretrained encoder model `resnetv1d50_b32x8_imagenet_20210531-db14775a.pth` from the 
celebrated repo [**mmclassification**](https://github.com/open-mmlab/mmclassification).
Place `resnetv1d50_b32x8_imagenet_20210531-db14775a.pth` in the folder `pretrained`.
Then just run `bash train.sh`.
Without bells and whistles, you will get the state-of-the-art model trained solely on this dataset!

We will upload the checkpoint with the performance reported in our paper soon latter.
To test our model on the testing set of AlphaMatting, just place the ckeckpoint in the folder `model` and run `bash test_alpha_matting.sh`.

Detailed descriptions will be updated soon!

If you have any question, please feel free to contact me!
