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
    # img.load_image()
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

    # calculate the size of the photo array
    size = (maxs - mins)
    centre = ((mins + maxs) / 2)

    # plot the centre of the photo array at the centre of the graph
    xspan = (centre[0] - (size[0] / 1.5), centre[0] + (size[0] / 1.5))
    yspan = (centre[1] - (size[1] / 1.5), centre[1] + (size[1] / 1.5))

    print("Centre: ", centre)
    print("Size: ", size)
    print("Xspan: ", xspan)
    print("yspan: ", yspan)


    # Creating plot

    xs = [x.down_angle.transformation_mat[0, 3] for x in imageGroups]
    ys = [x.down_angle.transformation_mat[1, 3] for x in imageGroups]
    zs = [x.down_angle.transformation_mat[2, 3] for x in imageGroups]

    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes()

    point_of_interest = (0, 0)
    circle1 = plt.Circle(point_of_interest, radius=50.0, color='r', fill=False)

    print("Creating 3D scatter graph from ", len(xs), "points")
    # ax.scatter(xs, ys, color='b')
    #
    # ax.add_patch(circle1)
    #
    # plt.xlim(xspan)
    # plt.ylim(yspan)
    # # ax.set_zlim(-1.5, 1.5)
    # plt.title("Drone Plots")
    # plt.show()

    pos = []
    ds = []
    annots = []
    ii = 0
    for grp in imageGroups:
        # for image in grp.images:
        #     if ii % 7 == 0:
        #         x, y, z = image.get_pos()
        #         x0s.append(x)
        #         y0s.append(y)
        #         z0s.append(z)
        #         d = image.get_image_vector()
        #         ds.append(d)
        #     ii += 1
        if ii % 1 == 0:
            img = grp.down_angle
            x, y, z = img.get_pos()
            pos.append((x, y, z))
            d = img.get_image_vector()
            ds.append(d)
            annots.append(img.image_id)
        ii += 1

    pos = np.asarray(pos)

    print(np.asarray(ds).shape)
    ds = np.asarray(ds)
    print(len(pos))
    print(len(ds))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    pos = pos[:55]
    ds = ds[:55]
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], ds[:, 0], ds[:, 1], ds[:, 2])
    for i in range(len(pos)):
        ax.text(pos[i, 0], pos[i, 1], pos[i, 2], annots[i], (1, 0, 0))
    # p = plt.Circle(point_of_interest, 50.0, fill=False)
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    ax.set_zlim(0, 400)
    plt.show()


if __name__ == '__main__':
    main()
