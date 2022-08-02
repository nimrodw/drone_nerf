import numpy as np
import matplotlib.pyplot as plt

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
