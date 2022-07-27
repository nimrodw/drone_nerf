import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from itertools import combinations

import json
colmap_frames = []
xml_frames = []
with open('other_transforms/colmap_transform.json') as json_file:
    data = json.load(json_file)
    colmap_frames = data["frames"]
with open('other_transforms/xml_4_transforms.json') as json_file:
    data = json.load(json_file)
    xml_frames = data["frames"]

# idea: take a bunch of colmap frames and compare them to the "raw value" that we get from the xml file.
xml_frames = sorted(xml_frames, key=lambda d: d['file_path'])
colmap_frames = sorted(colmap_frames, key=lambda d: d['file_path'])
# print(xml_frames)
# they're not necessarily ordered correctly.

xml_mat = np.array([])
for f in xml_frames:
    xml_mat = np.append(xml_mat, np.array(f["transform_matrix"]))
xml_mat = xml_mat.reshape(len(xml_frames), 4, 4)
colmap_mat = np.array([])
for f in colmap_frames:
    colmap_mat = np.append(colmap_mat, np.array(f["transform_matrix"]))
colmap_mat = colmap_mat.reshape(len(colmap_frames), 4, 4)
colmap_mat = colmap_mat[:400,:,:]


xml_mat = xml_mat[:, :3, 3]
# print(xml_mat)
colmap_mat = colmap_mat[:, :3, 3]
#put the z axis last
colmap_mat[:, [0, 2]] = colmap_mat[:, [2, 0]]
# print(colmap_mat)

print("Average distances between points")


def dist(p1, p2):
    if len(p1) == 4:
        x1, y1, z1, _ = p1
        x2, y2, z2, _ = p2
    else:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


mins = xml_mat.min(axis=0)
maxs = xml_mat.max(axis=0)
print("Drone, mins: ", mins)
print("Drone, maxs: ", maxs)

c_mins = colmap_mat.min(axis=0)
c_maxs = colmap_mat.max(axis=0)
print("COLMAP, mins: ", c_mins)
print("COLMAP, maxs: ", c_maxs)

size = maxs - mins
print("Size: ", size)
centre = -((mins+maxs) / 2)
print("CENTRE: ", centre)
scale = np.array([[2/size[0],0,0,0],
                 [0,2/size[1],0,0],
                 [0,0,1/size[2],0],
                 [0,0,0,1]])
translate = np.array([[1,0,0,centre[0]],
                     [0,1,0,centre[1]],
                     [0,0,1,centre[2]],
                     [0,0,0,1]])

matm = scale * translate
print("MatM: ", matm)
xml_mat = np.c_[xml_mat, np.ones(xml_mat.shape[0])]
scaled_drone = []
for x in xml_mat:
    n = matm @ x
    scaled_drone.append(n)
    print(n)
scaled_drone = np.asarray(scaled_drone)

print("Drone Metadata, std: ", scaled_drone.std(axis=0))
print("COLMAP Data, std: ", colmap_mat.std(axis=0))

print("Drone Metadata scaled, min: ", scaled_drone.min(axis=0))
print("Drone Metadata scaled, max: ", scaled_drone.max(axis=0))

distances = [dist(p1, p2) for p1, p2 in combinations(scaled_drone.tolist(), 2)]
avg_distance = sum(distances) / len(distances)
print("Average Drone Dist: ", avg_distance)

distances = [dist(p1, p2) for p1, p2 in combinations(colmap_mat.tolist(), 2)]
avg_distance = sum(distances) / len(distances)
print("Average colmap_mat Dist: ", avg_distance)

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(scaled_drone[:, 0], scaled_drone[:, 1], scaled_drone[:, 2], color='b')
ax.scatter3D(colmap_mat[:, 0], colmap_mat[:, 1], colmap_mat[:, 2], color='r')

plt.title("COLMAP points vs Drone points")

# show plot
plt.show()

plt.show()

