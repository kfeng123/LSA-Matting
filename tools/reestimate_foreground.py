import sys
import os
sys.path.insert(0,".")
import pymatting
import matting.utils.config as config
import numpy as np

#config.fg_path
#config.new_fg_path
if not os.path.exists(config.new_fg_path):
    os.makedirs(config.new_fg_path)

for f in os.listdir(config.fg_path):
    if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
        img = pymatting.load_image(os.path.join(config.fg_path, f))
        alpha = pymatting.load_image(os.path.join(config.alpha_path, f))
        foreground = pymatting.estimate_foreground_ml(alpha[:,:,np.newaxis]*img, alpha)
        foreground = (1 - (1-alpha[:,:,np.newaxis])**2 ) * img + (1-alpha[:,:,np.newaxis])**2 * foreground
        pymatting.save_image(os.path.join(config.new_fg_path, f), foreground)



