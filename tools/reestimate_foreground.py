import sys
import os
sys.path.insert(0,".")
import pymatting
import matting.utils.config as config
import numpy as np
import cv2

# reestimate the foreground using the method in ``A Closed-Form Solution to Natural Image Matting''
def bad_estimate():
    if not os.path.exists(config.bad_fg_path):
        os.makedirs(config.bad_fg_path)

    for f in os.listdir(config.fg_path):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
            img = pymatting.load_image(os.path.join(config.fg_path, f))
            alpha = pymatting.load_image(os.path.join(config.alpha_path, f))
            foreground = pymatting.estimate_foreground_cf(alpha[:,:,np.newaxis]*img, alpha)
            foreground = alpha[:,:,np.newaxis]  * img + (1-alpha[:,:,np.newaxis]) * foreground

            assert(len(os.path.splitext(f)) == 2)
            the_name = os.path.splitext(f)[0] + ".png"
            pymatting.save_image(os.path.join(config.bad_fg_path, the_name), foreground)



def my_refine_foreground_iteration(img, alpha, init_estimate, iter_num):
    h, w = alpha.shape
    alpha_pad = np.pad(alpha, ((1,1), (1,1)), 'reflect')
    alpha_shift_1 = alpha_pad[1:(h+1), 0:w]
    alpha_shift_2 = alpha_pad[1:(h+1), 2:(w+2)]
    alpha_shift_3 = alpha_pad[0:h, 1:(w+1)]
    alpha_shift_4 = alpha_pad[2:(h+2), 1:(w+1)]

    denominator = 4 * alpha ** 2 + 4 * (2 - alpha) - alpha_shift_1 - alpha_shift_2 - alpha_shift_3 - alpha_shift_4
    denominator = denominator[:,:, np.newaxis]

    term1 = (4 * alpha[:,:,np.newaxis] / denominator ) * img

    w1 = (2 - alpha - alpha_shift_1)[:,:,np.newaxis]
    w2 = (2 - alpha - alpha_shift_2)[:,:,np.newaxis]
    w3 = (2 - alpha - alpha_shift_3)[:,:,np.newaxis]
    w4 = (2 - alpha - alpha_shift_4)[:,:,np.newaxis]

    the_result = init_estimate
    for t in range(iter_num):
        the_result_pad = np.pad(the_result, ((1,1), (1,1), (0,0)), 'reflect')

        the_result = term1 + \
        (
            w1 * the_result_pad[1:(h+1), 0:w, :] +
            w2 * the_result_pad[1:(h+1), 2:(w+2), :] +
            w3 * the_result_pad[0:h, 1:(w+1), :] +
            w4 * the_result_pad[2:(h+2), 1:(w+1), :]
        ) / denominator
    return the_result

def my_refine_foreground(img, alpha, init_estimate):
    h, w = alpha.shape
    the_estimate = init_estimate
    for scale in range(5)[::-1]:
        scaled_img = cv2.resize(img, (w // 2**scale, h // 2**scale), interpolation = cv2.INTER_LINEAR )
        scaled_alpha = cv2.resize(alpha, (w // 2**scale, h // 2**scale), interpolation = cv2.INTER_LINEAR )
        the_estimate = cv2.resize(the_estimate, (w // 2**scale, h // 2**scale), interpolation = cv2.INTER_LINEAR )
        the_estimate = my_refine_foreground_iteration(scaled_img, scaled_alpha, the_estimate, 2**scale * 10)

    the_estimate = (np.clip(the_estimate, 0, 1) * 255).astype(np.uint8)
    return the_estimate

def new_estimate():
    if not os.path.exists(config.new_fg_path):
        os.makedirs(config.new_fg_path)

    for f in os.listdir(config.fg_path):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
            fg = cv2.imread(os.path.join(config.fg_path, f))[:,:,:3] / 255.
            alpha = cv2.imread(os.path.join(config.alpha_path, f), 0) / 255.
            foreground = my_refine_foreground(alpha[:,:,np.newaxis]*fg, alpha, fg)

            assert(len(os.path.splitext(f)) == 2)
            the_name = os.path.splitext(f)[0] + ".png"
            cv2.imwrite(os.path.join(config.new_fg_path, the_name), foreground)

if __name__ ==  "__main__":
    new_estimate()

    # do not run bad estimate, it will make a worse dataset. This is only for comparison
    #bad_estimate()


