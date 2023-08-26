import os
import uuid
from contextlib import ExitStack
from functools import wraps

import zmq
from zmq.decorators import _Decorator

__all__ = ['multi_socket', 'auto_bind']

class _MyDecorator(_Decorator):
    def __call__(self, *dec_args, **dec_kwargs):
        kw_name, dec_args, dec_kwargs = self.process_decorator_args(*dec_args, **dec_kwargs)
        num_socket_str = dec_kwargs.pop('num_socket')

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                num_socket = getattr(args[0], num_socket_str)
                targets = [self.get_target(*args, **kwargs) for _ in range(num_socket)]
                with ExitStack() as stack:
                    for target in targets:
                        obj = stack.enter_context(target(*dec_args, **dec_kwargs))
                        args = args + (obj,)

                    return func(*args, **kwargs)

            return wrapper

        return decorator

class _SocketDecorator(_MyDecorator):
    def process_decorator_args(self, *args, **kwargs):
        """Also grab context_name out of kwargs"""
        kw_name, args, kwargs = super(_SocketDecorator, self).process_decorator_args(*args, **kwargs)
        self.context_name = kwargs.pop('context_name', 'context')
        return kw_name, args, kwargs

    def get_target(self, *args, **kwargs):
        """Get context, based on call-time args"""
        context = self._get_context(*args, **kwargs)
        return context.socket

    def _get_context(self, *args, **kwargs):
        if self.context_name in kwargs:
            ctx = kwargs[self.context_name]

            if isinstance(ctx, zmq.Context):
                return ctx

        for arg in args:
            if isinstance(arg, zmq.Context):
                return arg
        # not specified by any decorator
        return zmq.Context.instance()


def multi_socket(*args, **kwargs):
    return _SocketDecorator()(*args, **kwargs)

def auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://127.0.0.1')
    else:
        # Get the location for tmp file for sockets
        try:
            tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
            if not os.path.exists(tmp_dir):
                raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
            tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
        except KeyError:
            tmp_dir = '*'
        socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')