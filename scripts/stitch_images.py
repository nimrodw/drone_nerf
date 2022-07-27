# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
	help="whether to crop out largest rectangular region")
ap.add_argument("-d", "--downsampling", type=float, default=1.0,
	help="downsampling sf")
args = vars(ap.parse_args())
# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
imagePaths = imagePaths[:30]
images = []
print("[INFO] Stitching ", len(imagePaths), " images")
# loop over the image paths, load each one, and add them to our
# images to stich list
downsampling = args["downsampling"]
print("[INFO] Applying a downsampling scale of ", downsampling)
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	height, width, layers = image.shape
	width = (int(width * downsampling))
	height = (int(height * downsampling))
	size = (width,height)
	resized = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
	images.append(resized)
# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
if status == 0:
	# write the output stitched image to disk
	cv2.imwrite(args["output"], stitched)
	# display the output stitched image to our screen
	#cv2.imshow("Stitched", stitched)
	#cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))
