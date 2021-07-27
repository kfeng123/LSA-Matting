sys.path.insert(0,".")
import pymatting
import matting.utils.config as config

#config.fg_path
#config.new_fg_path
if not os.path.exists(config.new_fg_path):
    os.makedirs(config.new_fg_path)

for f in os.listdir(mydir):
    if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
        img = pymatting.load_image(os.path.join(config.fg_path, f))
        alpha = pymatting.load_image(os.path.join(config.alpha_path, f))
        foreground = estimate_foreground_ml(img, alpha)
        cutout = stack_images(foreground, alpha)
        save_image(os.path.join(config.new_fg_path, f), cutout)



