import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms
import logging

#from scipy.ndimage import morphology

from PIL import Image

from ..utils import config

interpolation_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def random_resize(input, size):
    tmp = random.randint(0, len(interpolation_list)-1)
    return cv2.resize(input, size, interpolation = interpolation_list[tmp])

def random_warpAffine(input, rot_mat, size):
    tmp = random.randint(0, len(interpolation_list)-1)
    return cv2.warpAffine(input, rot_mat, size, flags=interpolation_list[tmp])

def generate_trimap_edt(fg, fg_wide):
    h, w = fg.shape[0], fg.shape[1]
    clicks = np.zeros((h,w,6))
    if(np.count_nonzero(fg) > 0):
        dt_mask = -cv2.distanceTransform(1 - fg, cv2.DIST_L2, 0)**2
        L = 320
        clicks[:, :, 0] = np.exp(dt_mask / (2 * ((0.02 * L)**2)))
        clicks[:, :, 1] = np.exp(dt_mask / (2 * ((0.08 * L)**2)))
        clicks[:, :, 2] = np.exp(dt_mask / (2 * ((0.16 * L)**2)))
    if(np.count_nonzero(fg_wide) > 0):
        dt_mask = -cv2.distanceTransform(1 - fg_wide, cv2.DIST_L2, 0)**2
        L = 320
        clicks[:, :, 3] = np.exp(dt_mask / (2 * ((0.02 * L)**2)))
        clicks[:, :, 4] = np.exp(dt_mask / (2 * ((0.08 * L)**2)))
        clicks[:, :, 5] = np.exp(dt_mask / (2 * ((0.16 * L)**2)))

    return clicks * 255

# take in a binary mask (np.uint8), output the Boundary-to-Pixel Direction as defined in the CVPR 2020 paper:
# Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation
def generate_BPD(mask):
    h, w = mask.shape[0], mask.shape[1]
    BPD = np.zeros((h, w, 2))
    if(np.count_nonzero(mask) > 0):
        if(mask.dtype != np.uint8):
            mask = mask.astype(np.uint8)
        mask_laplacian = cv2.Laplacian(1 - mask, cv2.CV_8U)
        mask_laplacian = np.clip(mask_laplacian, 0, 1)
        mask_laplacian_dt = cv2.distanceTransform(1- mask_laplacian, cv2.DIST_L2, 0)
        BPD[:,:,0] = cv2.Sobel(mask_laplacian_dt, cv2.CV_32F, 1, 0, ksize = 1)
        BPD[:,:,1] = cv2.Sobel(mask_laplacian_dt, cv2.CV_32F, 0, 1, ksize = 1)
        tmp = np.sqrt( BPD[:,:,0] ** 2 + BPD[:,:,1] ** 2 + 1e-6 )
        BPD[:,:,0] = BPD[:,:,0] / tmp
        BPD[:,:,1] = BPD[:,:,1] / tmp
        #BPD[:,:,0] = BPD[:,:,0] - 2 * BPD[:,:,0] * (mask > 0)
        #BPD[:,:,1] = BPD[:,:,1] - 2 * BPD[:,:,1] * (mask > 0)
    return BPD * 255

# take in a trimap (np.uint8), output the gradient of the SDF
#def generate_SDF_grad(trimap):
#    h, w = trimap.shape[0], trimap.shape[1]
#    SDF_grad = np.zeros((h, w, 2))
#    if(np.count_nonzero(mask) > 0):
#        if(mask.dtype != np.uint8):
#            mask = mask.astype(np.uint8)
#        mask_laplacian = cv2.Laplacian(1 - mask, cv2.CV_8U)
#        mask_laplacian = np.clip(mask_laplacian, 0, 1)
#        mask_laplacian_dt = cv2.distanceTransform(1- mask_laplacian, cv2.DIST_L2, 0)
#        BPD[:,:,0] = cv2.Sobel(mask_laplacian_dt, cv2.CV_32F, 1, 0, ksize = 1)
#        BPD[:,:,1] = cv2.Sobel(mask_laplacian_dt, cv2.CV_32F, 0, 1, ksize = 1)
#        tmp = np.sqrt( BPD[:,:,0] ** 2 + BPD[:,:,1] ** 2 + 1e-6 )
#        BPD[:,:,0] = BPD[:,:,0] / tmp
#        BPD[:,:,1] = BPD[:,:,1] / tmp
#        #BPD[:,:,0] = BPD[:,:,0] - 2 * BPD[:,:,0] * (mask > 0)
#        #BPD[:,:,1] = BPD[:,:,1] - 2 * BPD[:,:,1] * (mask > 0)
#    return BPD * 255

def generate_Urysohn_func(trimap):
    h, w = trimap.shape[0], trimap.shape[1]
    trimap_fg = (trimap > 200).astype(np.uint8)
    trimap_bg = (trimap < 50).astype(np.uint8)
    trimap_unknown = 1 - trimap_fg - trimap_bg

    trimap_fg_laplacian = cv2.Laplacian(trimap_fg, cv2.CV_8U)
    trimap_fg_laplacian = np.clip(trimap_fg_laplacian, 0, 1)

    trimap_bg_laplacian = cv2.Laplacian(trimap_bg, cv2.CV_8U)
    trimap_bg_laplacian = np.clip(trimap_bg_laplacian, 0, 1)

    nonzero_fg = np.count_nonzero(trimap_fg)
    nonzero_bg = np.count_nonzero(trimap_bg)

    if nonzero_fg == 0 and nonzero_bg == 0:
        my_trimap = np.ones((h, w)) * 0.5
    elif nonzero_fg > 0 and nonzero_bg == 0:
        my_trimap = np.ones((h, w)) * 0.5
    elif nonzero_fg == 0 and nonzero_bg > 0:
        my_trimap = np.ones((h, w)) * 0.5
    elif nonzero_fg > 0 and nonzero_bg > 0:
        trimap_fg_dt = cv2.distanceTransform(1-trimap_fg_laplacian, cv2.DIST_L2, 0).astype(np.float)
        trimap_bg_dt = cv2.distanceTransform(1-trimap_bg_laplacian, cv2.DIST_L2, 0).astype(np.float)
        my_trimap = trimap_bg_dt / (trimap_fg_dt + trimap_bg_dt + 1e-6)

    my_trimap = (2-my_trimap) * trimap_fg - my_trimap * trimap_bg + my_trimap * trimap_unknown
    my_trimap = (my_trimap + 1.) / 3.
    return my_trimap * 255.

#def generate_Urysohn_func(trimap):
#    h, w = trimap.shape[0], trimap.shape[1]
#    trimap_fg = (trimap > 200).astype(np.uint8)
#    trimap_bg = (trimap < 50).astype(np.uint8)
#    trimap_unknown = 1 - trimap_fg - trimap_bg
#
#    trimap_fg_laplacian = cv2.Laplacian(trimap_fg, cv2.CV_8U)
#    trimap_fg_laplacian = np.clip(trimap_fg_laplacian, 0, 1)
#
#    trimap_bg_laplacian = cv2.Laplacian(trimap_bg, cv2.CV_8U)
#    trimap_bg_laplacian = np.clip(trimap_bg_laplacian, 0, 1)
#
#    nonzero_fg = np.count_nonzero(trimap_fg)
#    nonzero_bg = np.count_nonzero(trimap_bg)
#
#    if nonzero_fg == 0 and nonzero_bg == 0:
#        my_trimap = np.ones((h, w)) * 0.5
#    elif nonzero_fg > 0 and nonzero_bg == 0:
#        my_trimap = np.ones((h, w)) * 0.5
#    elif nonzero_fg == 0 and nonzero_bg > 0:
#        my_trimap = np.ones((h, w)) * 0.5
#    elif nonzero_fg > 0 and nonzero_bg > 0:
#        trimap_fg_dt = cv2.distanceTransform(1-trimap_fg_laplacian, cv2.DIST_L2, 0).astype(np.float)
#        trimap_bg_dt = cv2.distanceTransform(1-trimap_bg_laplacian, cv2.DIST_L2, 0).astype(np.float)
#        my_trimap = trimap_bg_dt / (trimap_fg_dt + trimap_bg_dt + 1e-6)
#
#    my_trimap = 1 * trimap_fg + my_trimap * trimap_unknown
#    return my_trimap * 255.


def original_trimap(alpha):
    out = {}

    fg = np.equal(alpha, 255).astype(np.uint8)
    fg_wide = np.not_equal(alpha, 0).astype(np.uint8)
    unknown = fg_wide - fg

    if config.aux_loss_Urysohn:
        optimal_trimap = fg * 255
        optimal_trimap[unknown>0] = 128
        out['optimal_trimap_Urysohn'] = generate_Urysohn_func(optimal_trimap)

    #unknown = morphology.distance_transform_edt(unknown==0) <= np.random.randint(1, 21)
    distanceTrans = cv2.distanceTransform(1-unknown, cv2.DIST_L2, 0)
    unknown =  np.logical_or(np.logical_and(distanceTrans <= np.random.randint(1, 30), fg < 1),\
                np.logical_and(distanceTrans <= np.random.randint(1, 30), fg > 0) )
    trimap = fg * 255
    trimap[unknown] = 128

    if random.random() < 0.5:
        trimap = cv2.medianBlur(trimap, random.choice([3,5,7]))

    out['trimap'] = trimap.astype(np.uint8)

    if config.trimap_edt:
        out['trimap_edt'] = generate_trimap_edt((trimap > 200).astype(np.uint8), (trimap > 100).astype(np.uint8))

    if config.trimap_BPD:
        out['trimap_BPD'] = np.concatenate(
                ( generate_BPD((trimap > 200).astype(np.uint8)),  generate_BPD((trimap > 100).astype(np.uint8))  ),
                axis = 2)

    if config.trimap_Urysohn:
        out['trimap_Urysohn'] = generate_Urysohn_func(trimap)

    return out
    #if random.random() < 0.5:
    #    return trimap.astype(np.uint8)
    ## random erase
    #kernel_size = 101
    #half_size = kernel_size //2
    #h, w = alpha.shape
    #for i in range(3):
    #    ch = np.random.randint(half_size, h - half_size)
    #    cw = np.random.randint(half_size, w - half_size)
    #    trimap[(ch-half_size):(ch+half_size+1), (cw-half_size):(cw+half_size+1)] = 128
    #return trimap.astype(np.uint8)

class random_rotation:
    def __init__(self):
        self.degrees = [-180, 180]
        self.center_h_range = [-10, 10]
        self.center_w_range = [-10, 10]

    def __call__(self, alpha, fg, bg):
        h = fg.shape[0]
        w = fg.shape[1]

        # alpha and fg
        centerh = h//2 + random.randint(self.center_h_range[0], self.center_h_range[1])
        centerw = w//2 + random.randint(self.center_w_range[0], self.center_w_range[1])
        fg_degree = random.randint(self.degrees[0], self.degrees[1])
        rot_mat = cv2.getRotationMatrix2D((centerw, centerh), fg_degree, 1)

        #alpha = cv2.warpAffine(alpha, rot_mat, (w, h))
        alpha = random_warpAffine(alpha, rot_mat, (w, h))
        #fg = cv2.warpAffine(fg, rot_mat, (w, h))
        fg = random_warpAffine(fg, rot_mat, (w, h))

        # bg
        bg_centerh = h//2 + random.randint(self.center_h_range[0], self.center_h_range[1])
        bg_centerw = w//2 + random.randint(self.center_w_range[0], self.center_w_range[1])
        bg_degree = random.randint(self.degrees[0], self.degrees[1])
        bg_rot_mat = cv2.getRotationMatrix2D((bg_centerw, bg_centerh), bg_degree, 1)
        #bg = cv2.warpAffine(bg, bg_rot_mat, (w, h))
        bg = random_warpAffine(bg, bg_rot_mat, (w, h))

        return alpha, fg, bg

# standard gamma augmentation
#def gamma_aug(img, gamma):
#    return ( 255. * pow(img/255., gamma) ).clip(0, 255).astype(np.uint8)
# new gamma augmentation
def gamma_aug(img, gamma):
    if random.random() < 0.5:
        return ( 255. * pow(img/255., gamma) ).clip(0, 255).astype(np.uint8)
    else:
        return ( 255. * ( 1-pow(1-img/255., gamma) ) ).clip(0, 255).astype(np.uint8)

class random_crop(object):
    def __init__(self, flip=False):
        self.flip = flip

    def __call__(self, alpha, fg, bg, crop_h, crop_w):
        h, w = alpha.shape

        target = np.where((alpha > 0) & (alpha < 255))
        delta_h = center_h = crop_h / 2
        delta_w = center_w = crop_w / 2

        if len(target[0]) > 0:
            rand_ind = np.random.randint(len(target[0]))
            center_h = min(max(target[0][rand_ind], delta_h), h - delta_h)
            center_w = min(max(target[1][rand_ind], delta_w), w - delta_w)

        start_h = int(center_h - delta_h)
        start_w = int(center_w - delta_w)
        end_h   = int(center_h + delta_h)
        end_w   = int(center_w + delta_w)

        alpha  =alpha [start_h : end_h, start_w : end_w]
        fg    = fg   [start_h : end_h, start_w : end_w]
        bg    = bg   [start_h : end_h, start_w : end_w]

        # random flip
        if random.random() < 0.5:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)

        if random.random() < 0.5:
            bg = cv2.flip(bg, 1)

        return alpha, fg, bg

def get_files(mydir):
    res = []
    for f in os.listdir(mydir):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
            res.append(os.path.join(mydir, f))
    return res

class MatDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.logger = logging.getLogger("DeepImageMatting")

        self.args = args
        self.train_size_h = config.train_size_h
        self.train_size_w = config.train_size_w
        self.crop_size_h = config.crop_size_h
        self.crop_size_w = config.crop_size_w
        assert(len(self.crop_size_h) == len(self.crop_size_w))

        self.total_fg_list = []
        self.total_alpha_list = []
        self.total_bg_list = []

        # training data
        train_path_base = "/sda/WangRui/dataSets/Combined_Dataset/my_clean/train"
        fg_path = os.path.join(train_path_base, "fg")
        bg_path = os.path.join(train_path_base, "coco_bg")
        alpha_path = os.path.join(train_path_base, "alpha")
        for f in os.listdir(fg_path):
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
                self.total_fg_list.append(os.path.join(fg_path, f))
                self.total_alpha_list.append(os.path.join(alpha_path, f))
        for f in os.listdir(bg_path):
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
                self.total_bg_list.append(os.path.join(bg_path, f))

        # testing data
        test_path_base = "/sda/WangRui/dataSets/Combined_Dataset/my_clean/test"
        test_fg_path = os.path.join(test_path_base, "fg")
        test_alpha_path = os.path.join(test_path_base, "alpha")
        for f in os.listdir(test_fg_path):
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
                self.total_fg_list.append(os.path.join(test_fg_path, f))
                self.total_alpha_list.append(os.path.join(test_alpha_path, f))

        self.cnt = len(self.total_fg_list)
        assert(self.cnt > 0)

        for name in self.total_fg_list:
            assert(os.path.exists(name))
        for name in self.total_alpha_list:
            assert(os.path.exists(name))
        for name in self.total_bg_list:
            assert(os.path.exists(name))
        self.logger.info("MatDatasetOffline Samples: {}".format(self.cnt))

        self.random_rotation = random_rotation()
        self.random_crop = random_crop(flip=True)
        #self.ColorJitter = transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.2)
        self.ColorJitter = transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4)
        self.ToTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean = config.mean, std = config.std)

    def __len__(self):
        return len(self.total_bg_list)

    def __getitem__(self, index):
        #image_bg_path = os.path.join(self.bg_path, self.bg_list[index])
        image_bg_path = self.total_bg_list[index]

        #fg_index = index // 100
        #if random.random() < 0.3:
        #    fg_index = random.randint(0, len(self.bg_list) - 1 ) // 100
        #fg_index = random.randint(0, len(self.bg_list) - 1 ) // 100
        fg_index = random.randint(0, len(self.total_fg_list) - 1 )

        image_alpha_path = self.total_alpha_list[fg_index]
        image_fg_path = self.total_fg_list[fg_index]

        img_info = [image_fg_path, image_alpha_path, image_bg_path]

        # read fg, alpha
        alpha = cv2.imread(image_alpha_path, 0)
        fg = cv2.imread(image_fg_path)[:, :, :3]
        bg = cv2.imread(image_bg_path)[:, :, :3]

        # median blur for alpha
        if random.random() < 0.5:
            alpha = cv2.medianBlur(alpha, random.choice([3,5,7]))


        bh, bw, bc, = fg.shape
        img_info.append(fg.shape)

        bg = random_resize(bg, (bw, bh))

        #### random rotation
        if random.random() < 0.5:
            alpha, fg, bg = self.random_rotation(alpha, fg, bg)

        rand_ind = random.randint(0, len(self.crop_size_h) - 1)
        cur_crop_h = self.crop_size_h[rand_ind]
        cur_crop_w = self.crop_size_w[rand_ind]

        wratio = float(cur_crop_w) / bw
        hratio = float(cur_crop_h) / bh
        ratio = max(wratio, hratio)
        if ratio > 1:
            nbw = int(bw * ratio + 1.0)
            nbh = int(bh * ratio + 1.0)
            alpha = random_resize(alpha, (nbw, nbh))
            fg = random_resize(fg, (nbw, nbh))
            bg = random_resize(bg, (nbw, nbh))

        alpha, fg, bg = self.random_crop(alpha, fg, bg, cur_crop_h, cur_crop_w)

        if self.train_size_h != fg.shape[0] or self.train_size_w != fg.shape[1]:
            alpha = random_resize(alpha,  (self.train_size_w, self.train_size_h))
            fg    =random_resize(fg,     (self.train_size_w, self.train_size_h))
            bg    =random_resize(bg,     (self.train_size_w, self.train_size_h))

        # gamma augmentation
        if random.random() < 0.3:
            # gamma is from 0.5 to 2
            if random.random() < 0.5:
                fg = gamma_aug(fg, random.random()/2 + 0.5)
            else:
                fg = gamma_aug(fg, random.random() + 1.)
        if random.random() < 0.3:
            # gamma is from 0.5 to 2
            if random.random() < 0.5:
                bg = gamma_aug(bg, random.random()/2 + 0.5)
            else:
                bg = gamma_aug(bg, random.random() + 1.)

        if random.random() < 0.3:
            # gamma is from 0.5 to 2
            if random.random() < 0.5:
                alpha = gamma_aug(alpha, random.random()/2 + 0.5)
            else:
                alpha = gamma_aug(alpha, random.random() + 1.)



        trimap_dict = original_trimap(alpha)
        trimap = trimap_dict['trimap']
        ## debug
        #cv2.imwrite("result/debug/debug_{}_fg.png".format(index),fg)
        #cv2.imwrite("result/debug/debug_{}_bg.png".format(index),bg)
        #cv2.imwrite("result/debug/debug_{}_trimap.png".format(index),trimap)
        #cv2.imwrite("result/debug/debug_{}_trimap_edt0.png".format(index),trimap_dict['trimap_edt'][:,:,0])
        #cv2.imwrite("result/debug/debug_{}_trimap_edt1.png".format(index),trimap_dict['trimap_edt'][:,:,1])
        #cv2.imwrite("result/debug/debug_{}_trimap_edt2.png".format(index),trimap_dict['trimap_edt'][:,:,2])
        #cv2.imwrite("result/debug/debug_{}_trimap_edt3.png".format(index),trimap_dict['trimap_edt'][:,:,3])
        #cv2.imwrite("result/debug/debug_{}_trimap_edt4.png".format(index),trimap_dict['trimap_edt'][:,:,4])
        #cv2.imwrite("result/debug/debug_{}_trimap_edt5.png".format(index),trimap_dict['trimap_edt'][:,:,5])

        #########

        ###################################
        # blur
        #if random.random() < 0.3:
        #    t1 = random.randint(0,2)
        #    t2 = random.randint(0,2)
        #    fg = cv2.blur(fg, (2*t1 + 1, 2*t2+1))
        #    alpha = cv2.blur(alpha, (2*t1 + 1, 2*t2+1))
        # sharpen
        #if random.random() < 0.3:
        #    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #    fg = cv2.filter2D(fg, -1, kernel)
        #    alpha = cv2.filter2D(alpha, -1, kernel)
        ##################################

        fg_rgb = Image.fromarray(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB))
        fg_rgb = self.ColorJitter(fg_rgb)
        fg_norm = self.ToTensor(fg_rgb)

        ##################################
        # blur
        #if random.random() < 0.3:
        #    t1 = random.randint(0,2)
        #    t2 = random.randint(0,2)
        #    bg = cv2.blur(bg, (2*t1 + 1, 2* t2 + 1))
        # sharpen
        #if random.random() < 0.3:
        #    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #    bg = cv2.filter2D(bg, -1, kernel)
        ##################################

        bg_rgb = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        bg_rgb = self.ColorJitter(bg_rgb)
        bg_norm = self.ToTensor(bg_rgb)


        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])

        if config.trimap_edt:
            trimap_edt = torch.from_numpy(trimap_dict['trimap_edt']).permute(2,0,1).float()
            trimap = torch.cat([trimap, trimap_edt], 0)

        if config.trimap_BPD:
            trimap_BPD = torch.from_numpy(trimap_dict['trimap_BPD']).permute(2,0,1).float()
            trimap = torch.cat([trimap, trimap_BPD], 0)

        if config.trimap_Urysohn:
            trimap_Urysohn = torch.from_numpy(trimap_dict['trimap_Urysohn'])[None,:,:].float()
            trimap = torch.cat([trimap, trimap_Urysohn], 0)

        if config.aux_loss_Urysohn:
            optimal_trimap_Urysohn = torch.from_numpy(trimap_dict['optimal_trimap_Urysohn'])[None,:,:].float()

        #img_rgb = fg_norm.clone().detach()
        #for i in range(3):
        #    img_rgb[i,:,:] = fg_norm[i,:,:] * alpha/255. + bg_norm[i,:,:] * (1- alpha/255.)

        fg_norm = self.normalize(fg_norm)
        bg_norm = self.normalize(bg_norm)
        #img_norm = fg_norm * alpha/255. + bg_norm * (1- alpha/255.)

        #img_norm = self.normalize(img_rgb)

        if config.aux_loss_Urysohn:
            return fg_norm, bg_norm, alpha, trimap, optimal_trimap_Urysohn, img_info

        return fg_norm, bg_norm, alpha, trimap, img_info

