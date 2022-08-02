import numpy as np
import cv2
import multiprocessing
import scripts.generate_transforms_json
import camera as cam
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt




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
        self.transformation_mat = np.eye(4)

    def get_pos(self):
        return self.transformation_mat[:3, 3]

    def load_image(self):
        self.image_data = cv2.imread(self.image_path)
        self.sharpness = scripts.generate_transforms_json.sharpness(self.image_data)
        # why is the sharpness of higher quality images lower???

    def generate_homogenous_matrix(self, scale=1.0):
        sf = scripts.generate_transforms_json.scale(scale)
        # NeRF uses a coord system where Y = UP
        trans_mat = scripts.generate_transforms_json.translate_m(
            np.array([self.translation[0], self.translation[1], self.translation[2]]))
        self.transformation_mat = sf @ trans_mat @ self.transformation_mat

    def rotate_point(self):
        # this function rotates a point about itself
        # to do this, we have to move it to the origin, rotate, and then move it back
        centre = np.array([[1.0, 0, 0, -self.transformation_mat[0, 3]],
                           [0, 1.0, 0, -self.transformation_mat[1, 3]],
                           [0, 0, 1.0, -self.transformation_mat[2, 3]],
                           [0, 0, 0, 1.0]])
        self.transformation_mat = centre @ self.transformation_mat
        rx = scripts.generate_transforms_json.rot_x(self.rotation[0])
        ry = scripts.generate_transforms_json.rot_y(self.rotation[1])
        rz = scripts.generate_transforms_json.rot_z(self.rotation[2])
        self.transformation_mat = rx @ ry @ rz @ self.transformation_mat
        self.transformation_mat = -centre @ self.transformation_mat

    def scale_matrix(self, scale=1.0):
        sf = scripts.generate_transforms_json.scale(scale)
        self.transformation_mat = (sf @ self.transformation_mat)

    def generate_transform_matrix(self, average_position, sf):
        pos, rot = self.translation, self.rotation
        def Rx(theta):
            theta = theta * (np.pi / 180.)
            return np.matrix([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]])

        def Ry(theta):
            theta = theta * (np.pi / 180.)
            return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                              [0, 1, 0],
                              [-np.sin(theta), 0, np.cos(theta)]])

        def Rz(theta):
            theta = theta * (np.pi / 180.)
            return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])

        # R = Rz(rot[2]) * Ry(rot[1]) * Rx(rot[0])
        R = Rz(0.0) * Ry(0.0) * Rx(0.0)
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
        self.scale_matrix(sf)
        # self.rotate_point()


def produce_drone_image_list(xml_path="data/Xml/200_AT.xml"):
    droneImages = []
    xml_path = xml_path
    root = ET.parse(xml_path).getroot()
    photogroups = root.find("SpatialReferenceSystems/Block/Photogroups")

    positions = []
    rotations = []
    print("Loading Image Data from XML")
    camera_unset = True
    camera = None
    for pg in photogroups:
        if camera_unset:
            fl = float(pg.find('FocalLength').text)
            principal_point = [float(pg.find('PrincipalPoint/x').text), float(pg.find('PrincipalPoint/y').text)]
            distortion = [float(pg.find('Distortion/K1').text),
                          float(pg.find('Distortion/K2').text),
                          float(pg.find('Distortion/K3').text),
                          float(pg.find('Distortion/P1').text),
                          float(pg.find('Distortion/P2').text)]
            camera = cam.Camera(fl, principal_point, 1500.0, 1000.0, distortion)
            camera_unset = False

        pg_name = pg.find("Name").text
        photos = pg.findall("Photo")
        rots = []
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
            rots.append([k, phi, omega])
            image = droneImage(image_path, id, [x, y, z], [k, phi, omega])
            droneImages.append(image)
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)

    return droneImages, camera, positions, rotations
