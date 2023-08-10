import cv2
import numpy as np

from fdfat.utils.utils import LMK_PARTS, LMK_PART_NAMES

from fdfat.utils.pose_estimation import PoseEstimator
from fdfat.utils import box_utils
from fdfat.tracking import karman_filter

class Face:

    def __init__(self, bbox, landmark, frame_size, num_landmark=70):

        self.bbox = bbox.copy()
        self._bbox_stable = bbox.copy()
        self.landmark = landmark.copy()
        self._landmark_stable = landmark.copy()

        self.frame_width, self.frame_height = frame_size
        self.num_landmark = num_landmark

        self.pose_estimator = PoseEstimator(self.frame_width, self.frame_height)
        self.estimate_pose()

        self.kalman_filters = [karman_filter.create_filter() for _ in range(num_landmark)]
        for kal, (lmx, lmy) in zip(self.kalman_filters, landmark):
            kal.x = np.array([[lmx, 0, lmy, 0]]).T
            kal.P = np.eye(4) * 1000

    @property
    def stable_landmark(self):
        return self._landmark_stable

    @property
    def stable_bbox(self):
        return self._bbox_stable

    def update_bbox(self, bbox):
        self._bbox_stable = box_utils.stable_box(self._bbox_stable, bbox)
        self.bbox = bbox

    def update_ladnmark(self, landmark):
        stabled = []
        for kal, (lmx, lmy) in zip(self.kalman_filters, landmark):
            kal.predict()
            kal.update((lmx, lmy))
            x = kal.x
            stabled.append([x[0, 0], x[2, 0]])
        
        self.landmark = landmark
        self._landmark_stable = np.array(stabled)

        self.estimate_pose()

    def estimate_pose(self, stable=True):
        lmk = self.stable_landmark if stable else self.landmark
        self._pose = self.pose_estimator.solve(np.float32([(a[0], a[1]) for a in lmk[:68,:]]))

    def render(self, frame):
        sbox = self.stable_bbox

        lmk = self.stable_landmark.astype(np.int32)
        for begin, end in LMK_PARTS[:-1]:
            lx, ly = lmk[begin]
            for idx in range(begin+1, end):
                x, y = lmk[idx]
                cv2.line(frame, (lx, ly), (x, y), (255, 0, 0), 2)
                lx, ly = x, y

        for x, y in lmk:
            cv2.circle(frame, (x, y), 2, (255, 255, 255), 2)
            
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.rectangle(frame, (sbox[0], sbox[1]), (sbox[2], sbox[3]), (255, 0, 255), 4)

        self.pose_estimator.visualize(frame, self._pose)