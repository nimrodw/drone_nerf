import numpy as np
import cv2
import scripts.generate_transforms_json
import camera as cam
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.patches import Circle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d


class droneImage:

    def __init__(self, image_path, image_id, translation, rotation):
        # load the opencv image
        # load associated image params xyz + rot
        self.sharpness = None
        self.image_data = None
        self.image_path = image_path
        self.image_id = image_id
        self.translation = translation
        self.rotation = rotation  # k, phi, w
        self.transformation_mat = np.eye(4)

    def get_pos(self):
        return self.transformation_mat[:3, 3]

    def get_rot(self):
        return self.rotation

    def get_scale(self):
        scale = np.linalg.norm(self.transformation_mat[0, :3]), np.linalg.norm(
            self.transformation_mat[0, :3]), np.linalg.norm(self.transformation_mat[0, :3])
        return scale

    def load_image(self):
        self.image_data = cv2.imread(self.image_path)
        self.sharpness = scripts.generate_transforms_json.sharpness(self.image_data)
        # why is the sharpness of higher quality images lower???

    def generate_homogenous_matrix(self, scale=1.0):
        # NeRF uses a coord system where Y = UP
        sf = scripts.generate_transforms_json.scale(scale)
        trans_mat = scripts.generate_transforms_json.translate_m(
            np.array([self.translation[0], self.translation[1], self.translation[2]]))
        rot_mat = scripts.generate_transforms_json.rot_m(self.rotation)
        # sf @ rot_mat @
        self.transformation_mat = trans_mat @ self.transformation_mat

    def scale_matrix(self, scale=1.0):
        sf = scripts.generate_transforms_json.scale(scale)
        self.transformation_mat = (sf @ self.transformation_mat)
        return self.transformation_mat

    def get_image_vector(self, t=50):
        # equation of a circle: we need x,y,z and radius
        # cast a ray from the camera - r(t) = o + t*d (ray equals origin + length*vector)
        # x=x0+ta, y=y0+tb, z=z0+tc
        # a, b, c is direction, t is length, x0, y0, z0 is the origin of the ray
        k, phi, w = self.get_rot()

        w = w * (np.pi / 180.)
        phi = phi * (np.pi / 180.)
        k = k * (np.pi / 180.)
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(w), -np.sin(w)],
                          [0, np.sin(w), np.cos(w)]])
        rot_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
        rot_z = np.array([[np.cos(k), -np.sin(k), 0],
                          [np.sin(k), np.cos(k), 0],
                          [0, 0, 1]])
        rot_m = rot_x @ rot_y @ rot_z
        d = rot_m
        # we multiply the rotation matrix by the direction the camera array is facing
        # in this application, we can assume that the array faces down (negative on the z-axis)
        d = d @ np.array([0, 0, -1])
        d_hat = d / np.linalg.norm(d)
        return d_hat

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

        R = Rz(rot[2]) * Ry(rot[1]) * Rx(rot[0])
        # R = Rz(0.0) * Ry(0.0) * Rx(0.0)
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
        # xf = self.scale_matrix(1.0 / 550.0)
        xf = xf  # @ xf_rot
        self.transformation_mat = xf


def circle_2d_intersects_circle_2d(x1, y1, radius1, x2, y2, radius2):
    # does a given vector intersect a 2d object at x,y,z?
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes()
    # p1 = plt.Circle((x1, y1), radius1, fill=False)
    # ax.add_patch(p1)
    # p2 = plt.Circle((x2, y2), radius2, fill=False)
    # ax.add_patch(p2)
    # ax.scatter([x1, x2], [y1, y2], color='b')
    # plt.title("Plot Overlap")
    # plt.show()
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if dist <= radius1 + radius2:
        return True
    return False


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
            sensor_size = float(pg.find('SensorSize').text)
            camera = cam.Camera(fl, sensor_size, principal_point, 1500.0, 1000.0, distortion)
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
            omega = float(rotation.find('Omega').text)
            phi = float(rotation.find('Phi').text)
            k = float(rotation.find('Kappa').text)
            rotations.append([k, phi, omega])
            rots.append([k, phi, omega])
            image = droneImage(image_path, id, [x, y, z], [k, phi, omega])
            droneImages.append(image)
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)

    return droneImages, camera, positions, rotations
