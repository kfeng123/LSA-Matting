# IMPROVING DEEP IMAGE MATTING VIA LOCAL SMOOTHNESS ASSUMPTION
This is the official repository of [**IMPROVING DEEP IMAGE MATTING VIA LOCAL SMOOTHNESS ASSUMPTION**](https://arxiv.org/abs/2112.13809).
This repo includes all source codes (including data preprocessing code, training code and testing code). 
Have fun!

## Data preparation

We use the training data of [**Adobe Image Matting**](https://sites.google.com/view/deepimagematting) to train our model.
Please follow the instruction of Adobe Image Matting (AIM) to obtain the training foreground and background as well as the testing data.

Please modify the variable `train_path_base` in `matting/utils/config.py` such that the original AIM training foreground images are in the folder `train_path_base + "/fg"`, and place the background images in the folder `train_path_base + "/coco_bg"`, and place the ground truth alpha images in the folder `train_path_base + "/alpha"`.

Please modify the variable `test_path_base` in `matting/utils/config.py` to locate the AIM testing data (also called Composition-1k testing data) such that the testing images are in the folder `test_path_base + "/merged"`, and the testing trimaps are in the folder `test_path_base + "/trimaps"`, and the testing ground truth alphas are in the folder `test_path_base + "/alpha_copy"`.

## Foreground re-estimation

As described in our paper, the foreground of Adobe Image Matting can be improved to be more consistent with the local smoothness assumption.
To obtain the re-estimated foreground by our algorithm, just run `python tools/reestimate_foreground_final.py`.

## Training

To train the model,
first click [**here**](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth) 
to download the pretrained encoder model `resnetv1d50_b32x8_imagenet_20210531-db14775a.pth` from the 
celebrated repo [**mmclassification**](https://github.com/open-mmlab/mmclassification).
Place `resnetv1d50_b32x8_imagenet_20210531-db14775a.pth` in the folder `pretrained`.
Then just run `bash train.sh`.
Without bells and whistles, you will get the state-of-the-art model trained solely on this dataset!
By default, the model is trained for the 200 epochs.
Note that the reported results in our paper are the models trained for 100 epochs.
Thus, you have a great chance to obtain a better model than that reported in our paper!

## Testing

In this [**link**](https://cowtransfer.com/s/f0978719141847), we provide the checkpoint with best performance reported in our paper. 

To test our model on the Composition-1k testing data, please place the checkpoint in the folder `model`.
Please change the 105 line of the file `matting/models/model.py` to `for the_step in range(1)`. This modification in essense disables the backpropagating refinement, or else the testing process costs much time.
Then just run `bash test.sh`.

To test our model on the testing set of AlphaMatting, just place the checkpoint in the folder `model` and run `bash test_alpha_matting.sh`.

## Acknowledgments

If you use techniques in this project in your research, please cite our paper.
```
@misc{wang2021ImprovingDeepImageMatting,
      title={Improving Deep Image Matting Via Local Smoothness Assumption}, 
      author={Rui Wang and Jun Xie and Jiacheng Han and Dezhen Qi},
      year={2021},
      eprint={2112.13809},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you have any question, please feel free to raise issues!

Below I list some other open source (or partly open source) projects on image matting.
I learn a lot from these projects.
(For a more comprehensive list of projects on image matting, see [**wchstrife/Awesome-Image-Matting**](https://github.com/wchstrife/Awesome-Image-Matting).)
Thank you for sharing your codes!
I am proud to be one of you!

- [**foamliu/Deep-Image-Matting**](https://github.com/foamliu/Deep-Image-Matting)
- [**foamliu/Deep-Image-Matting-PyTorch**](https://github.com/foamliu/Deep-Image-Matting-PyTorch)
- [**huochaitiantang/pytorch-deep-image-matting**](https://github.com/huochaitiantang/pytorch-deep-image-matting)
- [**Yaoyi-Li/GCA-Matting**](https://github.com/Yaoyi-Li/GCA-Matting)
- [**poppinace/indexnet_matting**](https://github.com/poppinace/indexnet_matting)
- [**pymatting/pymatting**](https://github.com/pymatting/pymatting)
- [**MarcoForte/FBA_Matting**](https://github.com/MarcoForte/FBA_Matting) (FBAMatting is really awesome and influential! I am looking forward to the publication of this work! It truly deserves to be published in a top conference or journal! I am also looking forward to the fully open sourced version!)
