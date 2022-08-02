import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from itertools import combinations
import json
import xml.etree.ElementTree as ET
import os

# get raw position values
xml_path = "data/Xml/200_AT.xml"
root = ET.parse(xml_path).getroot()
photogroups = root.find("SpatialReferenceSystems/Block/Photogroups")

positions = []
rotations = []
print("Loading Image Data from XML")
camera_unset = True
for pg in photogroups:
    pg_name = pg.find("Name").text
    photos = pg.findall("Photo")
    for photo in photos:
        id = int(photo.find('Id').text)
        image_path = photo.find('ImagePath').text
        image_path = image_path.replace("LMY_PREFIX_PATH", "data/downsampled/_025")
        center = photo.find("Pose/Center")
        x = float(center.find('x').text)
        y = float(center.find('y').text)
        z = float(center.find('z').text)
        positions.append([x, y, z])

        rotation = photo.find("Pose/Rotation")
        k = float(rotation.find('Omega').text)
        phi = float(rotation.find('Phi').text)
        omega = float(rotation.find('Kappa').text)
        rotations.append([k, phi, omega])

positions = np.asarray(positions)
rotations = np.asarray(rotations)

print(positions.shape)
print("XML mins: ", np.min(positions, axis=0))
print("XML maxs: ", np.max(positions, axis=0))

raw_mat = []
for pos in positions:
    x, y, z = pos
    raw_mat.append(np.array([[1.0, 0, 0, x],
                             [0, 1.0, 0, y],
                             [0, 0, 1.0, z],
                             [0, 0, 0, 1.0]]))

raw_mat = np.asarray(raw_mat)

colmap_frames = []
xml_frames = []
with open('other_transforms/colmap_transform.json') as json_file:
    data = json.load(json_file)
    colmap_frames = data["frames"]
with open('other_transforms/xml_4_scaled_transforms.json') as json_file:
    data = json.load(json_file)
    xml_frames = data["frames"]

for f in xml_frames:
    f['file_path'] = os.path.basename(f['file_path'])
for f in colmap_frames:
    f['file_path'] = os.path.basename(f['file_path'])
# idea: take a bunch of colmap frames and compare them to the "raw value" that we get from the xml file.
xml_frames = sorted(xml_frames, key=lambda d: d['file_path'])
colmap_frames = sorted(colmap_frames, key=lambda d: d['file_path'])

filenames = ["4_00101.JPG", "4_00500.JPG", "4_00250.JPG"]
c_ = []
x_ = []
for c in colmap_frames:
    if c['file_path'] in filenames:
        c_.append(c['transform_matrix'])
for x in xml_frames:
    if x['file_path'] in filenames:
        x_.append(x['transform_matrix'])
# print(np.asarray(x_))
# print(np.asarray(c_))

# X * ? = C
# ? = inv(X) @ C
tr = np.linalg.inv(x_) @ c_

print(np.allclose(x_ @ tr, c_))
mean_tr = np.mean(tr, axis=0)

print(np.asarray(c_[2]))
print(x_[2] @ mean_tr)


xml_mat = np.array([])
for f in xml_frames:
    xml_mat = np.append(xml_mat, np.array(f["transform_matrix"]))
xml_mat = xml_mat.reshape(len(xml_frames), 4, 4)
colmap_mat = np.array([])
for f in colmap_frames:
    colmap_mat = np.append(colmap_mat, np.array(f["transform_matrix"]))
colmap_mat = colmap_mat.reshape(len(colmap_frames), 4, 4)
colmap_mat = colmap_mat[:400, :, :]

# Need to change the thing to get all the xml transformation

colmap_mat[:, 0, 3], colmap_mat[:, 1, 3] = colmap_mat[:, 1, 3], colmap_mat[:, 0, 3]


def dist(p1, p2):
    x1, y1, z1 = p1[0, 3], p1[1, 3], p1[2, 3]
    x2, y2, z2 = p2[0, 3], p2[1, 3], p2[2, 3]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


mins = xml_mat[:, :3, 3].min(axis=0)
maxs = xml_mat[:, :3, 3].max(axis=0)
# print("Drone, mins: ", mins)
# print("Drone, maxs: ", maxs)

c_mins = colmap_mat[:, :3, 3].min(axis=0)
c_maxs = colmap_mat[:, :3, 3].max(axis=0)
# print("COLMAP, mins: ", c_mins)
# print("COLMAP, maxs: ", c_maxs)

size = maxs - mins
# print("Size: ", size)
centre = -((mins + maxs) / 2)
# print("CENTRE: ", centre)
scale = np.array([[1 / size[0], 0, 0, 0],
                  [0, 1 / size[1], 0, 0],
                  [0, 0, 1 / size[2], 0],
                  [0, 0, 0, 1]])
translate = np.array([[1, 0, 0, centre[0]],
                      [0, 1, 0, centre[1]],
                      [0, 0, 1, centre[2]],
                      [0, 0, 0, 1]])

matm = np.eye(4) @ scale @ translate

# xml_mat = np.c_[xml_mat, np.ones(xml_mat.shape[0])]
scaled_drone = []
tr_drone = []
test = []
for x in xml_mat:
    n = matm @ x
    t = mean_tr @ x
    tr_drone.append(t)
    scaled_drone.append(n)
for r in raw_mat:
    r = matm @ r
    test.append(r)
test = np.asarray(test)
tr_drone = np.asarray(tr_drone)
scaled_drone = np.asarray(scaled_drone)

# print("Drone Metadata, std: ", scaled_drone.std(axis=0))
# print("COLMAP Data, std: ", colmap_mat.std(axis=0))
#
# print("Drone Metadata scaled, min: ", scaled_drone.min(axis=0))
# print("Drone Metadata scaled, max: ", scaled_drone.max(axis=0))

distances = [dist(p1, p2) for p1, p2 in combinations(scaled_drone, 2)]
avg_distance = sum(distances) / len(distances)
# print("Average Drone Dist: ", avg_distance)

distances = [dist(p1, p2) for p1, p2 in combinations(colmap_mat, 2)]
avg_distance = sum(distances) / len(distances)
# print("Average colmap_mat Dist: ", avg_distance)

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
# ax.scatter3D(tr_drone[:, 0], tr_drone[:, 1], tr_drone[:, 2], color='pink')
ax.scatter3D(scaled_drone[:, 0], scaled_drone[:, 1], scaled_drone[:, 2], color='b')
# ax.scatter3D(colmap_mat[:, 0], colmap_mat[:, 1], colmap_mat[:, 2], color='r')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
plt.title("COLMAP points vs Drone points")

# show plot
plt.show()
