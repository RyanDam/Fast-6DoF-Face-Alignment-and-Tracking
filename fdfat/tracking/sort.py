import numpy as np

from fdfat.tracking.face import Face
from fdfat.utils import box_utils

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
  
class SORT:

    def __init__(self, frame_size, iou_threshold=0.3, max_age=30, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
        self.frame_size = frame_size
        self._trackers = []

    @property
    def trackers(self):
        return self._trackers

    def match_detections_to_tracks(self, detections, trackers):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of :
            matches: [n, 2] [idx_det, idx_track]
            unmatched_detections: [n] [idx_det]
            unmatched_trackers: [n] [idx_track]
        """
        if(len(trackers) == 0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        iou_matrix = box_utils.iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def predict(self):
        for t in self._trackers:
            t.predict()

    def re_evaluate(self):
        results = []
        i = len(self._trackers)
        for trk in reversed(self._trackers):
            if (trk.time_since_update < 1) \
                and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                results.append(trk)
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self._trackers.pop(i)
        return results

    def update(self, detections, landmarks):
        ## detections: [x, y, xx, yy, score]
        self.frame_count += 1

        # predict next location of tracker
        self.predict()

        # build track bbox for matching
        trackers = np.zeros((len(self.trackers), 5))
        for idx, t in enumerate(self._trackers):
            box = t.stable_bbox
            trackers[idx, :] = [box[0], box[1], box[2], box[3], 0]

        # matching
        matched, unmatched_dets, unmatched_trks = self.match_detections_to_tracks(detections, trackers)

        # update matched track
        for idx_det, idx_trk in matched:
            self._trackers[idx_trk].update_bbox(detections[idx_det][:4])
            self._trackers[idx_trk].update_ladnmark(landmarks[idx_det])

        # create and initialise new trackers for unmatched detections
        for idx_det in unmatched_dets:
            trk = Face(detections[idx_det][:4], self.frame_size, landmark=landmarks[idx_det])
            self._trackers.append(trk)

        results = self.re_evaluate()
            
        return results
