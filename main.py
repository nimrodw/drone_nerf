import numpy as np
import cv2
import multiprocessing
import scripts.generate_transforms_json
import drone_images
import image_group
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

NUM_PHOTOGROUPS = 5


# load the images from the data and create the matrices
def do_img_load(img, avg):
    img.load_image()
    img.generate_homogenous_matrix(scale=1.0)
    return img


# could make this asynchronous for each of the five photo groups
def main():
    droneImages, cam, positions, rotations = drone_images.produce_drone_image_list("data/Xml/200_AT.xml")
    print("Loading ", str(len(droneImages)), " images")
    positions = np.asarray(positions)
    meanx, meany, meanz = np.min(positions[:, 0]), np.min(positions[:, 1]), np.min(
        positions[:, 2])
    avg = np.array([meanx, meany, meanz])
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    print("Raw Max: ", maxs)
    print("Raw Min: ", mins)
    print("Raw avg: ", avg)
    # loading the images takes a long time
    # let's speed it up by splitting it into multiple processes
    pool = multiprocessing.Pool(processes=16)
    processes = [pool.apply_async(do_img_load, args=(image, avg,)) for image in droneImages]
    droneImages = [p.get() for p in tqdm(processes)]

    NUM_IMAGES = int(len(droneImages) / NUM_PHOTOGROUPS)

    # now we need to group by file name
    # e.g. *_00001.JPG all together
    # 0 - Right, # 1 - Front, # 2 - Left, # 3 - Back, # 4 - Down
    imageGroups = []
    for j in range(NUM_IMAGES):
        right = droneImages[j]
        front = droneImages[1 * NUM_IMAGES + j]
        left = droneImages[2 * NUM_IMAGES + j]
        back = droneImages[3 * NUM_IMAGES + j]
        down = droneImages[4 * NUM_IMAGES + j]
        imggrp = image_group.ImageGroup(front, back, left, right, down)
        imageGroups.append(imggrp)
    print("There are ", len(imageGroups), " image groups")

    centre = []
    for grp in imageGroups:
        for img in grp.images:
            # don't rotate twice!!
            # img.generate_transform_matrix(avg, 1.0)
            p = img.get_pos()
            centre.append(p)
    centre = np.asarray(centre)
    centre = np.mean(centre, axis=0)
    print("Centre: ", centre)
    centre = centre.reshape(3,)
    print(imageGroups[80].down_angle.transformation_mat)
    positions = []
    # centre the mesh at 0, 0, 0
    for grp in imageGroups:
        for img in grp.images:
            translation = np.matrix([[1.0, 0, 0, -centre[0]],
                                     [0, 1.0, 0, -centre[1]],
                                     [0, 0, 1.0, -centre[2]],
                                     [0, 0, 0, 1.0]])
            img.transformation_mat = translation @ img.transformation_mat
            positions.append(img.get_pos())
    positions = np.asarray(positions)
    meanx, meany, meanz = np.min(positions[:, 0]), np.min(positions[:, 1]), np.min(
        positions[:, 2])
    avg = np.array([meanx, meany, meanz])
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    print("Raw Max: ", maxs)
    print("Raw Min: ", mins)
    print("Raw avg: ", avg)
    print(imageGroups[80].down_angle.transformation_mat)
    scale_factor = 0.001
    scripts.generate_transforms_json.export_to_json(cam, imageGroups,
                                                    "transforms.json", 1, scale=scale_factor, down_only=True)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    for grp in imageGroups:
        for img in grp.images:
            img.scale_matrix(scale_factor)

    xs = [x.down_angle.transformation_mat[0, 3] for x in imageGroups]
    ys = [x.down_angle.transformation_mat[1, 3] for x in imageGroups]
    zs = [x.down_angle.transformation_mat[2, 3] for x in imageGroups]
    ax.scatter3D(xs[:20], ys[:20], ys[:20], color='b')
    # ax.scatter3D(colmap_mat[:, 0], colmap_mat[:, 1], colmap_mat[:, 2], color='r')
    i = 0
    for grp in imageGroups:
        print(grp.down_angle.image_path, i)
        ax.text(xs[i], ys[i], zs[i], grp.down_angle.image_path, (1,0,0))
        i += 1
        if i > 20:
            break

    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)
    ax.set_zlim(-1.5, 1.5)
    plt.title("Drone Plots")
    plt.show()


if __name__ == '__main__':
    main()
