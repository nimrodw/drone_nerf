import cv2
import numpy as np
import glob

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def generate_video(photo_dir, out_name, FPS=15, downsampling=1.0):
    img_array = []
    for fname in sorted(glob.glob(photo_dir)):
        img = cv2.imread(fname)
        height, width, layers = img.shape
        width = (int(width * downsampling))
        height = (int(height * downsampling))
        size = (width,height)
        resized = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
        img_array.append(resized)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
    print("hello, mate")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print(out_name, " has finished...")


# generate_video("data/2_Amit/200M/Pic/2/*.JPG", "low_res_2.avi", FPS=5, downsampling=0.05)
# generate_video("data/2_Amit/200M/Pic/3/*.JPG", "low_res_3.avi", FPS=5, downsampling=0.05)
# generate_video("data/2_Amit/200M/Pic/4/*.JPG", "low_res_4.avi", FPS=5, downsampling=0.05)

#now I want to plot all the locations that the photos were taken from.
xs = np.array([])
ys = np.array([])
zs = np.array([])

root = ET.parse('../data/Xml/200_AT.xml').getroot()
photogroups = root.find("SpatialReferenceSystems/Block/Photogroups")
for pg in photogroups:
    pg_name = pg.find("Name").text
    print("Photogroup: ", pg_name)
    if pg_name == '0':
        photos = pg.findall("Photo")
        for photo in photos:
            center = photo.find("Pose/Center")
            x = float(center.find('x').text)
            y = float(center.find('y').text)
            z = float(center.find('z').text)
            xs = np.append(xs, [x])
            ys = np.append(ys, [y])
            zs = np.append(zs, [z])


label = np.arange(1,len(xs)+1, 1)
fig, ax = plt.subplots()

horiz = np.linspace(int(np.min(xs)), int(np.max(xs)), 20)
verts = np.linspace(int(np.min(ys))-1, int(np.max(ys))+1, 20)
ax.set_xticks(horiz)
ax.set_yticks(verts)
for i, txt in enumerate(label):
    ax.annotate(txt, (xs[i], ys[i]))
circle1 = plt.Circle((0, 0), 50.0, color='r', fill=False)


plt.plot(xs, ys, 'r+')
ax.add_patch(circle1)
plt.xlim(-250, 250)
plt.ylim(-250, 250)
fig.suptitle('Camera Positions of Photos (Photogroup 0)', fontweight ="bold")
plt.show()
