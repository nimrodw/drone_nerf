class Camera:

    def __init__(self, focal_length, principal_point, width, height, distortion):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.width = width
        self.height = height
        self.distortion = distortion