import cv2
import numpy as np
import glob
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input directories of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output directory")
ap.add_argument("-d", "--downsampling", type=float, default=0.5,
                help="downsampling sf")
args = vars(ap.parse_args())


# downsample images in directory and output them as an array of cv2 imags
def downsample_dir(photo_dir, downsampling):
    img_array = []
    name_array = []
    for fname in tqdm(sorted(glob.glob(photo_dir))):
        img = cv2.imread(fname)
        name_array.append(fname)
        height, width, layers = img.shape
        width = (int(width * downsampling))
        height = (int(height * downsampling))
        size = (width, height)
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img_array.append(resized)
    return img_array, name_array


def downsample_all_dirs(path, output_path, downsampling_factor=0.0125):
    # Get all the folders/directories in this path
    dirs = os.listdir(path)
    if dirs is not None:
        print("Creating destination folder " + output_path)
        try:
            os.mkdir(output_path)
        except FileExistsError:
            print("Directory already exists!")
            cont = input("Do you want to use this directory? (y/n)")
            if cont == 'n' or cont == 'N':
                print("Exiting...")
                return -1
        for dir in dirs:
            print("Downsampling all images in /" + dir + " with a downsampling factor of " + str(downsampling_factor))
            # now perform a downsample of all the images in each directory
            # copy them into the new dirs
            image_path = path + "/" + dir + "/*.JPG"
            img_array, name_array = downsample_dir(image_path, downsampling_factor)
            # save images
            new_path = output_path + "/" + dir
            try:
                os.mkdir(new_path)
            except FileExistsError:
                print(new_path + " already exists, continuing...")
            print("Writing downsampled images to " + new_path)
            for x in range(len(img_array)):
                cv2.imwrite(new_path + "/" + Path(name_array[x]).stem + ".JPG", img_array[x])


downsample_all_dirs(args['input'], args['output'], args['downsampling'])
