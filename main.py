import numpy as np
import cv2
import multiprocessing
import scripts.generate_transforms_json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
from sys import getsizeof

"""
This idea of this is to load all the image data into a python application
we then can construct the JSON data file which contains the transforms metadata for the images

"""


class Camera:

    def __init__(self, focal_length, principal_point, width, height, distortion):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.width = width
        self.height = height
        self.distortion = distortion


class droneImage:

    def __init__(self, image_path, image_id, translation, rotation):
        # load the opencv image
        # load associated image params xyz + rot
        self.sharpness = None
        self.image_data = None
        self.image_path = image_path
        self.image_id = image_id
        self.translation = translation
        self.rotation = rotation
        self.transformation_mat = []

    def load_image(self):
        self.image_data = cv2.imread(self.image_path)
        self.sharpness = scripts.generate_transforms_json.sharpness(self.image_data)
        # why is the sharpness of higher quality images lower???

    def generate_homogenous_matrix(self, scale_down=1.0):
        self.translation[0] /= scale_down
        self.translation[1] /= scale_down
        self.translation[2] /= scale_down
        rx = scripts.generate_transforms_json.rot_x(self.rotation[0])
        ry = scripts.generate_transforms_json.rot_y(self.rotation[1])
        rz = scripts.generate_transforms_json.rot_z(self.rotation[2])
        trans_mat = np.array([[1, 0, 0, self.translation[0]],
                              [0, 1, 0, self.translation[1]],
                              [0, 0, 1, self.translation[2]],
                              [0, 0, 0, 1]])
        self.transformation_mat = (np.identity(4) @ trans_mat @ rx @ ry @ rz)

    def generate_transform_matrix(self, average_position):
        pos, rot = self.translation, self.rotation
        def Rx(theta):
            return np.matrix([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]])

        def Ry(theta):
            return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                              [0, 1, 0],
                              [-np.sin(theta), 0, np.cos(theta)]])

        def Rz(theta):
            return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])

        R = Rz(rot[2]) * Ry(rot[1]) * Rx(rot[0])
        xf_rot = np.eye(4)
        xf_rot[:3, :3] = R

        xf_pos = np.eye(4)
        xf_pos[:3, 3] = pos - average_position

        # barbershop_mirros_hd_dense:
        # - camera plane is y+z plane, meaning: constant x-values
        # - cameras look to +x

        # Don't ask me...
        extra_xf = np.matrix([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])
        # NerF will cycle forward, so lets cycle backward.
        shift_coords = np.matrix([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])
        xf = shift_coords @ extra_xf @ xf_pos
        assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
        xf = xf @ xf_rot
        self.transformation_mat = xf


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
    # img.generate_transform_matrix(average_position=avg)
    img.generate_homogenous_matrix(scale_down=50.0)
    return img


# could make this asynchronous for each of the five photo groups
def main():

    droneImages = []
    xml_path = "data/Xml/200_AT.xml"
    root = ET.parse(xml_path).getroot()
    photogroups = root.find("SpatialReferenceSystems/Block/Photogroups")

    positions = []

    print("Loading Image Data from XML")
    camera_unset = True
    for pg in photogroups:
        if camera_unset:
            fl = float(pg.find('FocalLength').text)
            principal_point = [float(pg.find('PrincipalPoint/x').text), float(pg.find('PrincipalPoint/y').text)]
            distortion = [float(pg.find('Distortion/K1').text),
                          float(pg.find('Distortion/K2').text),
                          float(pg.find('Distortion/K3').text),
                          float(pg.find('Distortion/P1').text),
                          float(pg.find('Distortion/P2').text)]
            camera = Camera(fl, principal_point, 1500.0, 1000.0, distortion)
            camera_unset = False

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
            image = droneImage(image_path, id, [x, y, z], [k, phi, omega])
            droneImages.append(image)
    print("Loading ", str(len(droneImages)), " images")
    positions = np.asarray(positions)
    meanx, meany, meanz = np.min(positions[:, 0]), np.min(positions[:, 1]), np.min(
        positions[:, 2])
    avg = np.array([meanx, meany, meanz])

    pool = multiprocessing.Pool(5)
    processes = [pool.apply_async(do_img_load, args=(image,avg,)) for image in droneImages]
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
        front = droneImages[1*NUM_IMAGES+j]
        left = droneImages[2*NUM_IMAGES+j]
        back = droneImages[3*NUM_IMAGES+j]
        down = droneImages[4*NUM_IMAGES+j]
        imggrp = ImageGroup(front, back, left, right, down)
        imageGroups.append(imggrp)
    print("There are ", len(imageGroups), " image groups")

    # imageGroups = scripts.generate_transforms_json.whiten_positions(imageGroups)
    # imageGroups = scripts.generate_transforms_json.normalise_translation_mat(imageGroups)

    # imageGroups = scripts.generate_transforms_json.scale_to_unit_cube(imageGroups)

    scripts.generate_transforms_json.export_to_json(camera, imageGroups[200:350],
                                                    "transforms.json", 1)
    # scripts.generate_transforms_json.show_imagegroup_locations(imageGroups)
    print("END")


if __name__ == '__main__':
    main()
