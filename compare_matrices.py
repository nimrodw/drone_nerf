import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
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

print(xml_mat.shape)
print(colmap_mat.shape)

xml_mat = xml_mat[:, :3, 3]
# print(xml_mat)
colmap_mat = colmap_mat[:, :3, 3]
#put the z axis last
colmap_mat[:, [0, 2]] = colmap_mat[:, [2, 0]]
# print(colmap_mat)
mins = xml_mat.min(axis=0)
maxs = xml_mat.max(axis=0)
print("Drone, mins: ", mins)
print("Drone, maxs: ", maxs)

size = maxs - mins
centre = (mins+maxs) / 2
scale = np.array([[1/size[0],0,0,0],
                 [0,1/size[1],0,0],
                 [0,1/size[2],0,0],
                 [0,0,0,1]])
translate = np.array([[1,0,0,centre[0]],
                     [0,1,0,centre[1]],
                     [0,0,1,centre[2]],
                     [0,0,0,1]])

matm = scale * translate
print(matm)
xml_mat = np.c_[xml_mat, np.ones(xml_mat.shape[0])]

print(xml_mat[350] @ matm)
scaled_drone = xml_mat[:, ] @ matm


print("Drone Metadata, std: ", scaled_drone.std(axis=0))
print("COLMAP Data, std: ", colmap_mat.std(axis=0))

print("Drone Metadata, var: ", scaled_drone.var(axis=0))
print("COLMAP Data, var: ", colmap_mat.var(axis=0))

plt.scatter(scaled_drone[:, 0], scaled_drone[:,1], color='b')

plt.show()

