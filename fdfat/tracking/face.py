import cv2
import numpy as np

from fdfat.utils.utils import LMK_PARTS, LMK_PART_NAMES, get_color

from fdfat.utils.pose_estimation import PoseEstimator
from fdfat.utils import box_utils
from fdfat.tracking import karman_filter

class Face:

    counter = 0

    def __init__(self, bbox, frame_size, landmark=None, score=-1):
        self.frame_width, self.frame_height = frame_size

        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.face_score = score

        self.id = Face.counter
        Face.counter += 1

        self.bbox = bbox.copy()
        self._bbox_stable = bbox.copy()
        self.bbox_filter = karman_filter.create_bbox_filter(self.bbox)

        if landmark is not None:
            self._init_landmark(landmark)
        else:
            self.landmarked_initiated = False

    @property
    def stable_landmark(self):
        stabled = []
        for kal in self.landmark_filters:
            stabled.append([kal.x[0, 0], kal.x[2, 0]])
        return np.array(stabled)
        
    @property
    def stable_bbox(self):
        return karman_filter.convert_x_to_bbox(self.bbox_filter.x).reshape(-1)

    def _init_landmark(self, landmark):
        self.num_landmark = len(landmark)
        self.landmark = landmark.copy()
        self._landmark_stable = landmark.copy()

        self.landmark_filters = [
            karman_filter.create_point_filter(point) for _, point in zip(range(self.num_landmark), landmark)
        ]

        self.pose_estimator = PoseEstimator(self.frame_width, self.frame_height)
        self.estimate_pose()

        self.landmarked_initiated = True

    def predict(self):
        self.bbox_filter.predict()

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0

        self.time_since_update += 1

        self.history.append(self.stable_bbox)

    def update_bbox(self, bbox):

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        self.bbox_filter.update(karman_filter.convert_bbox_to_z(bbox))
        self.bbox = bbox

    def update_ladnmark(self, landmark):

        if not self.landmarked_initiated:
            self._init_landmark(landmark)
            return

        for kal, (lmx, lmy) in zip(self.landmark_filters, landmark):
            kal.predict()
            kal.update((lmx, lmy))
        
        self.landmark = landmark
        self.estimate_pose()

    def estimate_pose(self, stable=True):
        lmk = self.stable_landmark if stable else self.landmark
        self._pose = self.pose_estimator.solve(np.float32([(a[0], a[1]) for a in lmk[:68,:]]))

    def render(self, frame):

        tbox = self.bbox.astype(np.int32)
        cv2.rectangle(frame, (tbox[0], tbox[1]), (tbox[2], tbox[3]), (0,0,0), 4)

        sbox = self.stable_bbox.astype(np.int32)
        cv2.rectangle(frame, (sbox[0], sbox[1]), (sbox[2], sbox[3]), get_color(self.id), 4)

        if not self.landmarked_initiated:
            return

        lmk = self.stable_landmark.astype(np.int32)
        for begin, end in LMK_PARTS[:-1]:
            lx, ly = lmk[begin]
            for idx in range(begin+1, end):
                x, y = lmk[idx]
                cv2.line(frame, (lx, ly), (x, y), (255, 0, 0), 2)
                lx, ly = x, y

        for x, y in lmk:
            cv2.circle(frame, (x, y), 2, (255, 255, 255), 2)
            
        cv2.putText(frame, f"{self.id}", (sbox[0]+5, sbox[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.putText(frame, f"{self.face_score*100:0.1f}", (sbox[0]+5, sbox[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        self.pose_estimator.visualize(frame, self._pose)