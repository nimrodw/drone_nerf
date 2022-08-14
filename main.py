import glob
import os
import shutil

import numpy as np
import cv2
import multiprocessing

from mpl_toolkits.mplot3d import art3d

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

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(15)

    area_of_interest = (50, 50)
    area_radius = 100.0

    drone_height = 200
    pos = []
    ds = []
    annots = []
    # images that have been deemed to have a view of the area_of_interest that we would like to use
    choose_images = []
    ii = 0
    for grp in imageGroups:
        for img in grp.images:
            x, y, z = img.get_pos()
            # z_surface
            zs = z - drone_height
            t = z - zs

            d = img.get_image_vector(t=t)
            xs, ys = x + (t * d[0]), y + (t * d[1])

            field_w = (cam.sensor_size * drone_height) / cam.focal_length
            intersects = drone_images.circle_2d_intersects_circle_2d(xs, ys, field_w, area_of_interest[0],
                                                                     area_of_interest[1], area_radius)
            iou = drone_images.intersection_over_union(xs, ys, field_w, area_of_interest[0],
                                                       area_of_interest[1], area_radius)
            if intersects and iou > 0.35:
                print("x0, y0: ", x, y, " x1, y1", xs, ys, " d: ", d, " field_w: ", field_w)
                print("IoU: ", iou)
                choose_images.append(img.image_path)
                pos.append((x, y, z))
                annots.append(img.image_id)
                ds.append(t * d)
                p = plt.Circle((xs, ys), field_w, fill=False)
                ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=100, zdir="z")
            else:
                annots.append("")
            ii += 1

    print("Found ", len(choose_images), " images that can see the area of interest")
    if len(choose_images) > 15:
        with open("data/dst/image_paths.txt", "w") as file:
            file.truncate(0)
            existing_files = glob.glob('data/dst/images/*.JPG')
            for f in existing_files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error : %s: %s" % (f, e.strerror))
            for path in choose_images:
                file.write(path + "\n")
                shutil.copy(path, "data/dst/images/")

        pos = np.asarray(pos)

        print(np.asarray(ds).shape)
        ds = np.asarray(ds)

        ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], ds[:, 0], ds[:, 1], ds[:, 2])
        for i in range(pos.shape[0]):
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], annots[i], (1, 0, 0))
        p = plt.Circle(area_of_interest, area_radius, fill=False)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=100, zdir="z")
        plt.xlim(-400, 400)
        plt.ylim(-400, 400)
        ax.set_zlim(0, 400)
        plt.show()
    else:
        print("Not enough images!")


if __name__ == '__main__':
    main()
