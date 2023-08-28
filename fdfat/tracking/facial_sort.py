import cv2
import time
import numpy as np
from termcolor import colored
import zmq
import zmq.decorators as zmqd

from fdfat.nn.onnx import LandmarkAligner, FaceDetector
from fdfat.tracking.sort import SORT
from fdfat.utils import box_utils
from fdfat.utils import profiler

from fdfat.tracking.utils.worker import Worker
from fdfat.tracking.utils.protocol import *
from fdfat.tracking.utils.logger import *
from fdfat.tracking.utils.zmq_utils import *

def current_millis():
    return int(time.time()*1000)

class DetectorWorker(Worker):

    def get_model(self, args):
        detector = FaceDetector(args.track_detector)
        return detector
    
    def predict(self, model, img):
        boxes, probs = model.predict(img, threshold=0.75)

        detections = np.zeros((len(boxes), 5))
        for i, (b, s) in enumerate(zip(boxes, probs)):
            lb = box_utils.to_landmark_box(b)
            detections[i,:] = [lb[0], lb[1], lb[2], lb[3], s]

        return detections
    
class FacialSORT:

    def __init__(self, args):
        self.args = args
        self.logger = set_logger(colored('MAIN', 'red'))
        self.child_processes = []

        self.sort_tracker = None
        self.detector_wait_millis = 1000/self.args.track_detector_fps
        self.detector_last_push_stamp = -9999

        self.frame_id = 0
        self.current_faces = [] # to visualize

        self.landmark = LandmarkAligner(self.args.track_landmark)

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PAIR)
    def _run(self, _, to_det_socket, from_det_socket):

        self.logger.info('bind all sockets...')
        addr_detector = auto_bind(to_det_socket)
        self.logger.info('bind all sockets... DONE')

        self.logger.info('starting face detector...')
        proc_face_detect = DetectorWorker(self.args, 0, addr_detector, name="DETECTOR")
        self.child_processes.append(proc_face_detect)
        proc_face_detect.start()
        addr_from_face_detector = to_det_socket.recv().decode('ascii')
        from_det_socket.connect(addr_from_face_detector)
        self.logger.info('starting face detector... DONE')

        for p in self.child_processes:
            p.is_ready.wait()

        poller = zmq.Poller()
        poller.register(from_det_socket, zmq.POLLIN)

        self.logger.info('socket all set')

        self.logger.info('initiate video stream...')
        if self.args.track_source == "video":
            cap = cv2.VideoCapture(self.args.input)
            maximum_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.args.track_source == "camera":
            cap = cv2.VideoCapture(0)
            maximum_frame_count = -1
        else:
            raise AttributeError(f"track_source '{self.args.track_source}' is not supported")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.logger.info('initiate video stream... DONE')

        if self.args.track_save is not None:
            writer = cv2.VideoWriter(self.args.track_save, 
                         cv2.VideoWriter_fourcc(*'MJPG'), frame_fps, (frame_width, frame_height))


        self.sort_tracker = SORT((frame_width, frame_height), 
                                 iou_threshold=self.args.track_sort_iou_threshold, 
                                 max_age=self.args.track_max_age, 
                                 min_hits=self.args.track_min_hit)

        bench_read = profiler.Profile("READ")
        bench_process = profiler.Profile("PROCESS")
        bench_send = profiler.Profile("SEND")
        bench_recv = profiler.Profile("RECV")
        bench_detector = profiler.Profile("DETECTOR")
        bench_landmark = profiler.Profile("LANDMARK")
        bench_render   = profiler.Profile("RENDER")
        bench_dummy    = profiler.Profile("DUMPY")

        while cap.isOpened():

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            with bench_read:
                ret, frame = cap.read()

                if not ret:
                    continue
                
                self.frame_id += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.args.track_source == "video" and self.frame_id >= maximum_frame_count:
                    break

            with bench_process:

                current_time = current_millis()
                if (current_time - self.detector_last_push_stamp) > self.detector_wait_millis:
                    with bench_send:
                        send_object(to_det_socket, frame)
                    self.detector_last_push_stamp = current_time

                socks = dict(poller.poll(1))
                if socks and socks.get(from_det_socket) == zmq.POLLIN:
                    # self.logger.info(f'recv res from detector at frame {self.frame_id}')
                    with bench_recv:
                        detections = recv_object(from_det_socket)

                    dets, lmks, scores = [], [], []
                    for bbox in detections:
                        bbox = bbox[:4].astype(np.int32)
                        with bench_landmark:
                            lmk, face_cls = self.landmark.predict_frame(frame, bbox, have_face_cls=True)

                        if face_cls >= 0.0:
                            lmk_box = box_utils.bbox_from_landmark(lmk).flatten()
                            dets.append(lmk_box)
                            lmks.append(lmk)
                            scores.append(face_cls)

                    det_track = np.zeros((len(dets), 5))
                    for i, (b, s) in enumerate(zip(dets, scores)):
                        det_track[i,:] = [*b, s]

                    self.current_faces = self.sort_tracker.update(det_track, lmks)
                else:
                    for f in self.sort_tracker.trackers:

                        f.predict()

                        with bench_landmark:
                            lmk, face_cls = self.landmark.predict_frame(frame, f.stable_bbox.astype(np.int32), have_face_cls=True)

                        if face_cls >= 0.5:
                            lmk_box = box_utils.bbox_from_landmark(lmk).flatten()
                            f.update_bbox(lmk_box)
                            f.update_ladnmark(lmk)

                        f.face_score = face_cls

                    self.current_faces = self.sort_tracker.re_evaluate()

            for f in self.current_faces:
                with bench_render:
                    f.render(frame)
            fps = int(1000/(bench_process.report()+1e-6))
            cv2.putText(frame, f"FPS: {fps}", (5, 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            if self.args.track_visualize:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Render', frame)
            
            if self.args.track_save is not None:
                writer.write(frame)

            # print("====")
            # bench_read.print_report()
            # bench_process.print_report()
            # bench_send.print_report()
            # bench_recv.print_report()
            # bench_detector.print_report()
            # bench_landmark.print_report()
            # bench_render.print_report()
            # bench_dummy.print_report()
            # time.sleep(0.1)

            # time.sleep(0.005)
            
        cap.release()
        if self.args.track_save is not None:
            writer.release()
        if self.args.track_visualize:
            cv2.destroyAllWindows()

        for p in self.child_processes:
            p.close()

        self.logger.info('terminated!')
