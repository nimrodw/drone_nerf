import numpy as np
import cv2
import matplotlib.pyplot as plt
import json


def translate_m(translation):
    return np.array([[1, 0, 0, translation[0]],
                     [0, 1, 0, translation[1]],
                     [0, 0, 1, translation[2]],
                     [0, 0, 0, 1]])


def rot_m(rots):
    return rot_z(rots[2]) * rot_y(rots[1]) * rot_x(rots[0])


def rot_x(rot):
    rot = rot * (np.pi / 180.)
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(rot), np.sin(rot), 0],
                     [0, -np.sin(rot), np.cos(rot), 0],
                     [0, 0, 0, 1]])


def rot_y(rot):
    rot = rot * (np.pi / 180.)
    return np.array([[np.cos(rot), 0, -np.sin(rot), 0],
                     [0, 1, 0, 0],
                     [np.sin(rot), 0, np.cos(rot), 0],
                     [0, 0, 0, 1]])


def rot_z(rot):
    rot = rot * (np.pi / 180.)
    return np.array([[np.cos(rot), -np.sin(rot), 0, 0],
                     [np.sin(rot), np.cos(rot), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def rot_x_opk(rot):
    rot = rot * (np.pi / 180.)
    return np.array([[1, 0, 0],
                     [0, np.cos(rot), -np.sin(rot)],
                     [0, np.sin(rot), np.cos(rot)]])


def scale(sf):
    return np.array([[sf, 0, 0, 0],
                     [0, sf, 0, 0],
                     [0, 0, sf, 0],
                     [0, 0, 0, 1]])


# this is the method that Nvidia's instant NGP uses to calculate the sharpness of an image
# peculiarly, lower resolution images seem to have a higher sharpness
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def show_imagegroup_locations(image_array):
    xs = np.array([])
    ys = np.array([])
    zs = np.array([])
    for images in image_array:
        translation = images.avg_position
        xs = np.append(xs, [translation[0]])
        ys = np.append(ys, [translation[1]])
        zs = np.append(zs, [translation[2]])
    label = np.arange(1, len(xs) + 1, 1)
    fig, ax = plt.subplots()
    # this is a comment
    horiz = np.linspace(int(np.min(xs)), int(np.max(xs)), 20)
    verts = np.linspace(int(np.min(ys)) - 1, int(np.max(ys)) + 1, 20)
    ax.set_xticks(horiz)
    ax.set_yticks(verts)
    for i, txt in enumerate(label):
        ax.annotate(txt, (xs[i], ys[i]))
    plt.plot(xs, ys, 'r+')
    fig.suptitle('Camera Positions of Photos', fontweight="bold")
    plt.show()


def whiten_positions(imageGroup):
    # whiten positions
    # subtract mean and divide by std
    translations = []
    for imggrp in imageGroup:
        images = imggrp.images
        translations.append([i.transformation_mat for i in images])
    translations = np.array(translations)

    meanx, meany, meanz = np.mean(translations[:, :, 0, 3]), np.mean(translations[:, :, 1, 3]), np.mean(
        translations[:, :, 2, 3])
    stdx, stdy, stdz = np.std(translations[:, :, 0, 3]), np.std(translations[:, :, 1, 3]), np.std(
        translations[:, :, 2, 3])
    print("means: ", meanx, meany, meanz)
    print("stds: ", stdx, stdy, stdz)

    before = np.empty((0, 3), int)
    after = np.empty((0, 3), int)
    for imggrp in imageGroup:
        images = imggrp.images
        for img in images:
            x, y, z = img.transformation_mat[0, 3], img.transformation_mat[1, 3], img.transformation_mat[2, 3]
            before = np.vstack((before, np.array([x, y, z])))
            x_hat = (x - meanx) / stdx
            y_hat = (y - meany) / stdy
            z_hat = (z - meanz) / stdz
            after = np.vstack((after, np.array([x_hat, y_hat, z_hat])))
            img.transformation_mat[0, 3], img.transformation_mat[1, 3], img.transformation_mat[
                2, 3] = x_hat, y_hat, z_hat
    return imageGroup


def normalise_translation_mat(imageGroup):
    # normalise translations
    translations = []
    for imggrp in imageGroup:
        images = imggrp.images
        translations.append([i.transformation_mat for i in images])
    translations = np.array(translations)

    minx, miny, minz = np.min(translations[:, :, 0, 3]), np.min(translations[:, :, 1, 3]), np.min(
        translations[:, :, 2, 3])
    maxx, maxy, maxz = np.max(translations[:, :, 0, 3]), np.max(translations[:, :, 1, 3]), np.max(
        translations[:, :, 2, 3])
    print("mins: ", minx, miny, minz)
    print("maxs: ", maxx, maxy, maxz)

    before = np.empty((0, 3), int)
    after = np.empty((0, 3), int)
    for imggrp in imageGroup:
        images = imggrp.images
        for img in images:
            x, y, z = img.transformation_mat[0, 3], img.transformation_mat[1, 3], img.transformation_mat[2, 3]
            before = np.vstack((before, np.array([x, y, z])))
            x_hat = (x - minx) / (maxx - minx)
            y_hat = (y - miny) / (maxy - miny)
            z_hat = (z - minz) / (maxz - minz)
            after = np.vstack((after, np.array([x_hat, y_hat, z_hat])))
            img.transformation_mat[0, 3], img.transformation_mat[1, 3], img.transformation_mat[
                2, 3] = x_hat, y_hat, z_hat
    return imageGroup


def scale_to_unit_cube(imageGroup):
    # normalise translations
    translations = []
    for imggrp in imageGroup:
        images = imggrp.images
        translations.append([i.transformation_mat for i in images])

    translations = np.array(translations)
    minx, miny, minz = np.min(translations[:, :, 0, 3]), np.min(translations[:, :, 1, 3]), np.min(
        translations[:, :, 2, 3])
    maxx, maxy, maxz = np.max(translations[:, :, 0, 3]), np.max(translations[:, :, 1, 3]), np.max(
        translations[:, :, 2, 3])
    mins = np.array([minx, miny, minz])
    maxs = np.array([maxx, maxy, maxz])
    print("mins: ", mins)
    print("maxs: ", maxs)
    size = maxs - mins
    print("size: ", size)
    centre = (mins + maxs) / 2
    print("centre: ", centre)
    scale = np.array([[1 / size[0], 0, 0, 0],
                      [0, 1 / size[1], 0, 0],
                      [0, 0, 1 / size[2], 0],
                      [0, 0, 0, 1]])
    translate = np.array([[1, 0, 0, centre[0]],
                          [0, 1, 0, centre[1]],
                          [0, 0, 1, centre[2]],
                          [0, 0, 0, 1]])
    matm = scale * translate
    print(matm)
    for imggrp in imageGroup:
        images = imggrp.images
        for img in images:
            x, y, z = img.transformation_mat[0, 3], img.transformation_mat[1, 3], img.transformation_mat[2, 3]
            pos = np.array([x, y, z, 1.0]) @ matm
            img.transformation_mat[0, 3], img.transformation_mat[1, 3], img.transformation_mat[
                2, 3] = pos[0], pos[1], pos[2]
    return imageGroup


def scale_to_unit_cube_2(imageGroup):
    # unit cubise translations
    pos = []
    rot = []
    for imggrp in imageGroup:
        images = imggrp.images
        pos.append([i.translation for i in images])
        rot.append([i.rotation for i in images])

    pos = np.asarray(pos)
    rot = np.asarray(rot)
    pos = pos.reshape(len(imageGroup) * 5, 3)
    rot = rot.reshape(len(imageGroup) * 5, 3)

    mins = np.min(pos, axis=0)
    maxs = np.max(pos, axis=0)
    print("mins: ", mins)
    print("maxs: ", maxs)
    size = (maxs - mins)
    print("size: ", size)
    centre = -((mins + maxs) / 2)
    print("centre: ", centre)
    scale = np.array([[1 / size[0], 0, 0, 0],
                      [0, 1 / size[1], 0, 0],
                      [0, 0, 1 / size[2], 0],
                      [0, 0, 0, 1]])
    translate = np.array([[1, 0, 0, centre[0]],
                          [0, 1, 0, centre[1]],
                          [0, 0, 1, centre[2]],
                          [0, 0, 0, 1]])
    matm = np.eye(4) @ scale @ translate
    for imggrp in imageGroup:
        images = imggrp.images
        for img in images:
            img.transformation_mat = matm @ np.array([[1.0, 0, 0, img.translation[0]],
                                                      [0, 1.0, 0, img.translation[1]],
                                                      [0, 0, 1.0, img.translation[2]],
                                                      [0, 0, 0, 1.0]])
    return imageGroup


# convert the image group classes into a JSON that Instant NGP accepts
# "choose_num" is a parameter the essentially chooses how many to skip
# the function only chooses every "choose_num"th image group (set of five images)
def export_to_json(camera, image_groups, dst_path, choose_num, scale=0.002, down_only=False):
    frames = []
    i = 0
    if not down_only:
        for group in image_groups:
            if i % choose_num == 0:
                for img in group.images:
                    frame = {
                        "file_path": img.image_path,
                        "sharpness": img.sharpness,
                        "transform_matrix": img.transformation_mat.tolist()
                    }
                    frames.append(frame)
            i += 1
    else:
        for group in image_groups:
            if i % choose_num == 0:
                frame = {
                    "file_path": group.down_angle.image_path,
                    "sharpness": group.down_angle.sharpness,
                    "transform_matrix": group.down_angle.transformation_mat.tolist()
                }
                frames.append(frame)
            i += 1
    distortion = camera.distortion
    fl = camera.focal_length
    w, h = camera.width, camera.height
    dictionary = {
        "camera_angle_x": 0.86,
        "camera_angle_y": 0.59,
        # "fl_x": fl * 100,
        # "fl_y": fl * 100,
        "fl_x": 2535.7774397566544,  # * 0.0029,
        "fl_y": 2535.7774397566544,  # * 0.0029,
        "k1": distortion[0],
        "k2": distortion[1],
        "p1": distortion[3],
        "p2": distortion[4],
        "cx": w / 2,
        "cy": h / 2,
        "w": w,
        "h": h,
        "aabb_scale": 4,
        "scale": scale,
        "frames": frames
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open(dst_path, "w") as outfile:
        outfile.write(json_object)
    print("JSON File created at ", dst_path)
