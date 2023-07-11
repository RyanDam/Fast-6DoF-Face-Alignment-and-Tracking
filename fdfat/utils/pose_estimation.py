"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np

MODEL_3D_POINTS = [
    -73.393523 ,
    -72.775014 ,
    -70.533638 ,
    -66.850058 ,
    -59.790187 ,
    -48.368973 ,
    -34.121101 ,
    -17.875411 ,
    0.098749 ,
    17.477031 ,
    32.648966 ,
    46.372358 ,
    57.343480 ,
    64.388482 ,
    68.212038 ,
    70.486405 ,
    71.375822 ,
    -61.119406 ,
    -51.287588 ,
    -37.804800 ,
    -24.022754 ,
    -11.635713 ,
    12.056636 ,
    25.106256 ,
    38.338588 ,
    51.191007 ,
    60.053851 ,
    0.653940 ,
    0.804809 ,
    0.992204 ,
    1.226783 ,
    -14.772472 ,
    -7.180239 ,
    0.555920 ,
    8.272499 ,
    15.214351 ,
    -46.047290 ,
    -37.674688 ,
    -27.883856 ,
    -19.648268 ,
    -28.272965 ,
    -38.082418 ,
    19.265868 ,
    27.894191 ,
    37.437529 ,
    45.170805 ,
    38.196454 ,
    28.764989 ,
    -28.916267 ,
    -17.533194 ,
    -6.684590 ,
    0.381001 ,
    8.375443 ,
    18.876618 ,
    28.794412 ,
    19.057574 ,
    8.956375 ,
    0.381549 ,
    -7.428895 ,
    -18.160634 ,
    -24.377490 ,
    -6.897633 ,
    0.340663 ,
    8.444722 ,
    24.474473 ,
    8.449166 ,
    0.205322 ,
    -7.198266 ,
    -29.801432 ,
    -10.949766 ,
    7.929818 ,
    26.074280 ,
    42.564390 ,
    56.481080 ,
    67.246992 ,
    75.056892 ,
    77.061286 ,
    74.758448 ,
    66.929021 ,
    56.311389 ,
    42.419126 ,
    25.455880 ,
    6.990805 ,
    -11.666193 ,
    -30.365191 ,
    -49.361602 ,
    -58.769795 ,
    -61.996155 ,
    -61.033399 ,
    -56.686759 ,
    -57.391033 ,
    -61.902186 ,
    -62.777713 ,
    -59.302347 ,
    -50.190255 ,
    -42.193790 ,
    -30.993721 ,
    -19.944596 ,
    -8.414541 ,
    2.598255 ,
    4.751589 ,
    6.562900 ,
    4.661005 ,
    2.643046 ,
    -37.471411 ,
    -42.730510 ,
    -42.711517 ,
    -36.754742 ,
    -35.134493 ,
    -34.919043 ,
    -37.032306 ,
    -43.342445 ,
    -43.110822 ,
    -38.086515 ,
    -35.532024 ,
    -35.484289 ,
    28.612716 ,
    22.172187 ,
    19.029051 ,
    20.721118 ,
    19.035460 ,
    22.394109 ,
    28.079924 ,
    36.298248 ,
    39.634575 ,
    40.395647 ,
    39.836405 ,
    36.677899 ,
    28.677771 ,
    25.475976 ,
    26.014269 ,
    25.326198 ,
    28.323008 ,
    30.596216 ,
    31.408738 ,
    30.844876 ,
    47.667532 ,
    45.909403 ,
    44.842580 ,
    43.141114 ,
    38.635298 ,
    30.750622 ,
    18.456453 ,
    3.609035 ,
    -0.881698 ,
    5.181201 ,
    19.176563 ,
    30.770570 ,
    37.628629 ,
    40.886309 ,
    42.281449 ,
    44.142567 ,
    47.140426 ,
    14.254422 ,
    7.268147 ,
    0.442051 ,
    -6.606501 ,
    -11.967398 ,
    -12.051204 ,
    -7.315098 ,
    -1.022953 ,
    5.349435 ,
    11.615746 ,
    -13.380835 ,
    -21.150853 ,
    -29.284036 ,
    -36.948060 ,
    -20.132003 ,
    -23.536684 ,
    -25.944448 ,
    -23.695741 ,
    -20.858157 ,
    7.037989 ,
    3.021217 ,
    1.353629 ,
    -0.111088 ,
    -0.147273 ,
    1.476612 ,
    -0.665746 ,
    0.247660 ,
    1.696435 ,
    4.894163 ,
    0.282961 ,
    -1.172675 ,
    -2.240310 ,
    -15.934335 ,
    -22.611355 ,
    -23.748437 ,
    -22.721995 ,
    -15.610679 ,
    -3.217393 ,
    -14.987997 ,
    -22.554245 ,
    -23.591626 ,
    -22.406106 ,
    -15.121907 ,
    -4.785684 ,
    -20.893742 ,
    -22.220479 ,
    -21.025520 ,
    -5.712776 ,
    -20.671489 ,
    -21.903670 ,
    -20.328022 ,
]

class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image_width, image_height):
        """Init a pose estimator.

        Args:
            image_width (int): input image width
            image_height (int): input image height
        """
        self.size = (image_height, image_width)
        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

    def _get_full_model_points(self):
        """Get all 68 3D model points from file"""
        model_points = np.array(MODEL_3D_POINTS, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def solve(self, points):
        """Solve pose with all the 68 image points
        Args:
            points (np.ndarray): points on image.

        Returns:
            Tuple: (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def visualize(self, image, pose, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        rotation_vector, translation_vector = pose
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axes(self, img, pose):
        R, t = pose
        img = cv2.drawFrameAxes(img, self.camera_matrix,
                                self.dist_coeefs, R, t, 30)

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()
