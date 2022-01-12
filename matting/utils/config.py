import os
cuda = True

# data
input_format = "RGB"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# train data
train_path_base = "/sda/WangRui/dataSets/Combined_Dataset/my_clean/train"
fg_path = os.path.join(train_path_base, "fg")
new_fg_path = os.path.join(train_path_base, "kappa_fg")
#bad_fg_path = os.path.join(train_path_base, "bad_fg")
bg_path = os.path.join(train_path_base, "coco_bg")
alpha_path = os.path.join(train_path_base, "alpha")

#fg_list = open(os.path.join(train_path_base, "training_fg_names.txt")).readlines()
#fg_list = [name.strip() for name in fg_list]
#bg_list = open(os.path.join(train_path_base, "training_bg_names.txt")).readlines()
#bg_list = [name.strip() for name in bg_list]

# test data
test_path_base = "/sda/WangRui/dataSets/Combined_Dataset/my_clean/test"
test_img_path = os.path.join(test_path_base, "merged")
test_trimap_path = os.path.join(test_path_base, "trimaps")
test_alpha_path = os.path.join(test_path_base, "alpha_copy")

# fine-tuning data (training data of alphamatting.com)
#fine_tuning_names = ["GT" + str(i).zfill(2) + ".png" for i in range(1,28)]
#fine_tuning_img_list =[os.path.join("/sda/WangRui/alphamatting_train/input_training_lowres", name) for name in fine_tuning_names]
#fine_tuning_alpha_list =[os.path.join("/sda/WangRui/alphamatting_train/gt_training_lowres", name) for name in fine_tuning_names]
#fine_tuning_trimap1_list =[os.path.join("/sda/WangRui/alphamatting_train/trimap_training_lowres/Trimap1", name) for name in fine_tuning_names]
#fine_tuning_trimap2_list =[os.path.join("/sda/WangRui/alphamatting_train/trimap_training_lowres/Trimap2", name) for name in fine_tuning_names]

# train
batchSize = 16
threads = 8
saveDir = "model"
printFreq = 20
ckptSaveFreq = 10
testFreq= 10

#train_size_h = 512
#train_size_w = 512
train_size_h = 640
train_size_w = 640

#crop_size_h = [i for i in range(448, 577)]
#crop_size_w = [i for i in range(448, 577)]
crop_size_h = [i for i in range(480, 801)]
crop_size_w = [i for i in range(480, 801)]

# training data loader
total_fg_list = []
total_alpha_list = []
total_bg_list = []
for f in os.listdir(new_fg_path):
    if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
        total_fg_list.append(os.path.join(new_fg_path, f))
        alpha_name = os.path.splitext(f)[0] + ".jpg"
        total_alpha_list.append(os.path.join(alpha_path, alpha_name))
for f in os.listdir(bg_path):
    if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
        total_bg_list.append(os.path.join(bg_path, f))
train_add_test = False
if train_add_test:
    test_fg_path = os.path.join(test_path_base, "fg")
    test_origin_alpha_path = os.path.join(test_path_base, "alpha")
    for f in os.listdir(test_fg_path):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
            total_fg_list.append(os.path.join(test_fg_path, f))
            total_alpha_list.append(os.path.join(test_origin_alpha_path, f))

# test
max_size = 2176
if_test_grad = False
if_test_connect = False

# optimizer
nEpochs = 200
opt_method = "Adam"
lr = 5e-5
weight_decay = 1e-4
