# clac edge
import os
from edge_detector import *
from PIL import Image
# import torch
from image_transform import ImageTransform
from config import *
# from matplotlib import pyplot as plt
# import glob
# import numpy as np
import argparse

    
'''
python extract_edges.py --img_dir  --dest_dir
img_dir = './dog-breed-identification/test/'
dest_dir = './dog-breed-identification/testedge/'
'''



def main(img_dir, dest_dir):
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    im_tran = ImageTransform((HEIGHT, WIDTH), MEAN, STD)
        
    files = os.listdir(img_dir)
    for img_name in files:
        print(img_name)
        if not img_name.endswith('.jpg'): continue

        fullname = os.path.join(img_dir, img_name)
        im = Image.open(fullname)

        im = im_tran(im, 'train')
        edge_map = detect_edge_new(im[:3].permute(1,2,0)) # make it XxYx3!!!
        savename = os.path.join(dest_dir, img_name)
        Image.fromarray(edge_map).save(savename)
    

def mainSobel(img_dir, dest_dir):
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    im_tran = ImageTransform((HEIGHT, WIDTH), MEAN, STD)
        
    files = os.listdir(img_dir)
    for img_name in files:
        print(img_name)
        if not img_name.endswith('.jpg'): continue

        fullname = os.path.join(img_dir, img_name)
        im = Image.open(fullname)

        im = im_tran(im, 'train')
        edge_map = compute_energy_matrix(im[:3].permute(1,2,0)) # make it XxYx3!!!
        savename = os.path.join(dest_dir, img_name)
        Image.fromarray(edge_map).save(savename)
    

    
# for img_name in glob.glob(img_dir+'*.jpg'):            
#     file, ext = os.path.splitext(infile)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Edge Extraction on a Folder')
    parser.add_argument('--indir', type=str, help='Input dir for images')
    parser.add_argument('--outdir', type=str, help='Output dir for edges')
    args = parser.parse_args()
#     import pdb; pdb.set_trace()
#     print(args.indir)
    
    mainSobel(args.indir, args.outdir)
    print('Done!')
    
    # main(sys.argv[1], float(sys.argv[2])    