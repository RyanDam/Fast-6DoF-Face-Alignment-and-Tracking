import zmq
import zmq.decorators as zmqd
import multiprocessing
from multiprocessing import Process
from termcolor import colored

from .protocol import *
from .zmq_utils import multi_socket, auto_bind
from .logger import set_logger

class Worker(Process):

    def __init__(self, id, source_address, name='WORKER', color='yellow'):
        super().__init__()

        self.name = name
        self.color = color
        self.worker_id = id
        
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.source_address = source_address
        
        self.is_ready = multiprocessing.Event()

        self.logger = set_logger(colored('%s-%d' % (self.name, self.worker_id), self.color))

    def close(self):
        self.logger.info('Shutting down...')
        self.exit_flag.set()
        self.is_ready.clear()
        self.terminate()
        self.join()
        self.logger.info('Terminated!')

    def run(self):
        self._run()

    def get_model(self):
        raise NotImplementedError("'get_model' function is not implemented")
    
    def predict(self, model, img):
        raise NotImplementedError("'predict' function is not implemented")

    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PAIR)
    def _run(self, recv_socket, send_socket):
        logger = set_logger(colored('%s-%d' % (self.name, self.worker_id), self.color))
        logger.info("Starting worker {}...".format(self.worker_id))

        recv_socket.connect(self.main_thread_addr)
        addr_to_source = auto_bind(send_socket)
        recv_socket.send(addr_to_source.encode('ascii'))
        logger.info("Starting worker {}... DONE".format(self.worker_id))

        logger.info("Init model...")
        model = self.get_model()
        logger.info("Init model... DONE")

        self.is_ready.set()

        while not self.exit_flag.is_set():
            try:
                img = recv_object(recv_socket)
                res = self.predict(model, img)
                send_object(send_socket, res)
            except Exception as e:
                import traceback
                tb=traceback.format_exc()
                logger.error('{}\n{}'.format(e, tb))