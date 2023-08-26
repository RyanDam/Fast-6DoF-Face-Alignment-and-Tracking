import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)

def set_logger(context, logger_dir=None, verbose=True):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if verbose:
        formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d]: %(levelname)-.1s:' + context + ': %(message)s', datefmt=
        '%y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d]: %(levelname)-.1s:' + context + ': %(message)s', datefmt=
        '%y-%m-%d %H:%M:%S')
    
    if logger_dir:
        file_name = os.path.join(logger_dir, 'TTSServer_{:%Y-%m-%d}.log'.format(datetime.now()))
        handler = RotatingFileHandler(file_name, mode='a', maxBytes=10*1024*1024, backupCount=10, encoding=None, delay=0)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


