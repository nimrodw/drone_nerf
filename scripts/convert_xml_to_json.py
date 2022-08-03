import numpy as np
import json
import matplotlib.pyplot as plt

# read
file_path = "../other_transforms/colmap_transform.json"

with open(file_path) as json_file:
    data = json.load(json_file)
    xml_frames = data["frames"]

# for f in xml_frames:
#     f['file_path'] = os.path.basename(f['file_path'])

# idea: take a bunch of colmap frames and compare them to the "raw value" that we get from the xml file.
xml_frames = sorted(xml_frames, key=lambda d: d['file_path'])

# process
noise_factor = 1e-4
print(xml_frames[5]['transform_matrix'])
for x in xml_frames:
    t = np.asmatrix(x['transform_matrix'])
    t = t + np.random.normal(0, .02, (t.shape))
    x['transform_matrix'] = t.tolist()
print(xml_frames[5]['transform_matrix'])

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
# Creating plot
ax.scatter3D([np.asmatrix(x['transform_matrix'])[0, 3] for x in xml_frames],
             [np.asmatrix(x['transform_matrix'])[1, 3] for x in xml_frames],
             [np.asmatrix(x['transform_matrix'])[2, 3] for x in xml_frames], color='b')
plt.title("Drone Plots")
plt.show()


dictionary = {
    "camera_angle_x": 0.8643960223807634,
    "camera_angle_y": 0.5961654556998522,
    "fl_x": 1625.8962760681159,
    "fl_y": 1627.4094371437486,
    "k1": -0.04077603438461113,
    "k2": 0.048963739048656356,
    "p1": 0.0004865713443102642,
    "p2": 2.557346477425589e-05,
    "cx": 753.3040564356664,
    "cy": 494.2012044820192,
    "w": 1500.0,
    "h": 1000.0,
    "aabb_scale": 4,
    "frames": xml_frames
}
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
dst = "../other_transforms/noise_transform.json"
with open(dst, "w") as outfile:
    outfile.write(json_object)
