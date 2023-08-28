import cv2
import numpy as np
import onnxruntime as ort

from fdfat.utils import box_utils

# ONNX_BACKENDS = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
# ONNX_BACKENDS = ['CoreMLExecutionProvider']
ONNX_BACKENDS = ['CPUExecutionProvider']

class ONNXModel:

    def __init__(self, model_path, channel_first=True):

        self.model_path = model_path
        # self.input_width, self.input_height = input_size

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options, providers=ONNX_BACKENDS)

        if channel_first:
            _, _, self.input_height, self.input_width = self.session.get_inputs()[0].shape
        else:
            _, self.input_height, self.input_width, _ = self.session.get_inputs()[0].shape

    def preprocess(self, img):

        img = cv2.resize(img, (self.input_width, self.input_height))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128

        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return img
    
class FaceDetector(ONNXModel):

    def postprocess(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def predict(self, ori_img, threshold=0.5):
        
        height, width, _ = ori_img.shape

        # stat(ori_img)
        img = self.preprocess(ori_img)
        # stat(img)
        confidences, boxes = self.session.run([], {"input": img})

        boxes, _, probs = self.postprocess(width, height, confidences, boxes, threshold)

        return boxes, probs
    
class LandmarkAligner(ONNXModel):

    def predict(self, ori_img, have_face_cls=False):
        height, width, _ = ori_img.shape

        img = self.preprocess(ori_img)
        lmk = self.session.run([], {'input': img})[0]

        if have_face_cls:
            lmk, face_cls = lmk[0][:70*2].reshape((70,2)), lmk[0][70*2]
        else:
            lmk = lmk[0][:70*2].reshape((70,2))

        lmk += 0.5
        lmk[:,0] *= width
        lmk[:,1] *= height

        if have_face_cls:
            return lmk, face_cls
        
        return lmk
    
    def predict_frame(self, frame, bbox, have_face_cls=False):

        fheight, fwidth, _ = frame.shape
        lmk_box = box_utils.guard_bbox_inside(bbox, fwidth, fheight)

        face_img = frame[lmk_box[1]:lmk_box[3], lmk_box[0]:lmk_box[2], :]
        lmk = self.predict(face_img, have_face_cls=have_face_cls)

        if have_face_cls:
            lmk, face_cls = lmk

        lmk[:, 0] += lmk_box[0]
        lmk[:, 1] += lmk_box[1]

        if have_face_cls:
            return lmk, face_cls
        
        return lmk