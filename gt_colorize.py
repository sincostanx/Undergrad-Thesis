import glob
import os

from option import args
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

test_dir = "/home/yy/worameth/Desktop/nyu_depth_v2/official_splits/test/"
out_dir = "./gt_colorized/"
all_files = glob.glob(os.path.join(test_dir, "**/*.png"), recursive=True)

os.makedirs(out_dir, exist_ok=True)
plt.figure(figsize=(20,10))
for f in tqdm(all_files):
    basename = os.path.basename(f).split('.')[0]
    
    gt = np.asarray(Image.open(f), dtype=np.float32)
    gt = np.expand_dims(gt, axis=2)
    gt = gt / 1000.
    gt[gt < args.min_depth_eval] = args.min_depth
    gt[gt > args.max_depth_eval] = args.max_depth
    gt[np.isinf(gt)] = args.max_depth
    gt[np.isnan(gt)] = args.min_depth
    
    save_path = os.path.join(out_dir, basename + ".png")
    plt.clf()
    plt.imshow(gt, vmin=args.min_depth, vmax=args.max_depth)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        
out_dir = "./input_rescaled/"
all_files = glob.glob(os.path.join(test_dir, "**/*.jpg"), recursive=True)

os.makedirs(out_dir, exist_ok=True)
plt.figure(figsize=(20,10))
for f in tqdm(all_files):
    basename = os.path.basename(f).split('.')[0]
    
    img = np.asarray(Image.open(f), dtype=np.float32) / 255.
    
    save_path = os.path.join(out_dir, basename + ".jpg")
    plt.clf()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)