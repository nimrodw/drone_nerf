import numpy as np
import cv2
import multiprocessing
import scripts.generate_transforms_json
import drone_images
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from sys import getsizeof


# Class ImageGroup
# a collection of 5 droneImage classes, the five images are the
# images that were taken at the same moment
# this should give us a simple way to choose which images to use
class ImageGroup:

    def __init__(self, front_angle, back_angle, left_angle, right_angle, down_angle):
        self.images = [front_angle, back_angle, left_angle, right_angle, down_angle]
        self.front_angle = self.images[0]
        self.back_angle = self.images[1]
        self.left_angle = self.images[2]
        self.right_angle = self.images[3]
        self.down_angle = self.images[4]
        self.avg_position = np.mean((front_angle.translation,
                                     back_angle.translation,
                                     left_angle.translation,
                                     right_angle.translation,
                                     down_angle.translation), axis=0)

    def print_paths(self):
        for img in self.images:
            print(img.image_path)

    def scale_images(self, scale):
        for img in self.images:
            img.scale_matrix(scale)

    def display_images(self):
        fig = plt.figure(figsize=(12, 12))
        columns = 3
        rows = 3
        image = self.front_angle.image_data
        ax = fig.add_subplot(rows, columns, 2)
        ax.title.set_text(self.front_angle.image_path)
        plt.imshow(image)
        image = self.left_angle.image_data
        ax = fig.add_subplot(rows, columns, 4)
        ax.title.set_text(self.left_angle.image_path)
        plt.imshow(image)
        image = self.down_angle.image_data
        ax = fig.add_subplot(rows, columns, 5)
        ax.title.set_text(self.down_angle.image_path)
        plt.imshow(image)
        image = self.right_angle.image_data
        ax = fig.add_subplot(rows, columns, 6)
        ax.title.set_text(self.right_angle.image_path)
        plt.imshow(image)
        image = self.back_angle.image_data
        ax = fig.add_subplot(rows, columns, 8)
        ax.title.set_text(self.back_angle.image_path)
        plt.imshow(image)

        plt.show()


NUM_PHOTOGROUPS = 5


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

    pool = multiprocessing.Pool(5)
    processes = [pool.apply_async(do_img_load, args=(image, avg,)) for image in droneImages]
    droneImages = [p.get() for p in tqdm(processes)]

    NUM_IMAGES = int(len(droneImages) / NUM_PHOTOGROUPS)

    # now we need to group by file name
    # e.g. *_00001.JPG all together
    # 0 - Right
    # 1 - Front
    # 2 - Left
    # 3 - Back
    # 4 - Down
    imageGroups = []
    for j in range(NUM_IMAGES):
        right = droneImages[j]
        front = droneImages[1 * NUM_IMAGES + j]
        left = droneImages[2 * NUM_IMAGES + j]
        back = droneImages[3 * NUM_IMAGES + j]
        down = droneImages[4 * NUM_IMAGES + j]
        imggrp = ImageGroup(front, back, left, right, down)
        imageGroups.append(imggrp)
    print("There are ", len(imageGroups), " image groups")

    # imageGroups = scripts.generate_transforms_json.whiten_positions(imageGroups)
    # imageGroups = scripts.generate_transforms_json.normalise_translation_mat(imageGroups)



    # REMINDER - trying to do the scaling better!
    # imageGroups = scripts.generate_transforms_json.scale_to_unit_cube(imageGroups)
    #
    for grp in imageGroups:
        for img in grp.images:
            img.generate_transform_matrix(avg, 1./760.)

    # [x.down_angle.rotate_point() for x in imageGroups]
    for x in imageGroups:
        print(x.down_angle.rotation)

    # def plot_drone_points(imageGroup):
    #     pos = [x.down_angle.get_pos() for x in imageGroup]
    #     rot = [[1, 0, 0] for x in imageGroup]
    #     rot = np.asarray(rot)
    #     ax = plt.figure().add_subplot(projection='3d')
    #     ax.quiver(pos[0], pos[1], pos[2], rot[:, 0], rot[:, 1], rot[:, 2], normalize=True)
    #     plt.show()
    # plot_drone_points(imageGroups)
    scripts.generate_transforms_json.export_to_json(camera, imageGroups,
                                                    "transforms.json", 1, down_only=True)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D([x.down_angle.transformation_mat[0, 3] for x in imageGroups],
                 [x.down_angle.transformation_mat[1, 3] for x in imageGroups],
                 [x.down_angle.transformation_mat[2, 3] for x in imageGroups], color='b')
    # ax.scatter3D(colmap_mat[:, 0], colmap_mat[:, 1], colmap_mat[:, 2], color='r')

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    plt.title("Drone Plots")
    plt.show()

    # scripts.generate_transforms_json.show_imagegroup_locations(imageGroups)
    print("Running NERF")
    # os.system("./home/bertie/Documents/instant-ngp/build/testbed --scene /home/bertie/PycharmProjects/drone_nerf")


if __name__ == '__main__':
    main()
