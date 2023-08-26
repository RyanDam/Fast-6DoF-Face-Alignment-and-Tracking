import numpy as np
import zlib, pickle
import zmq
from zmq.utils import jsonapi

def send_to_next_raw(msg, msg_info, dst, flags=0, copy=True, track=False):
    dst.send_multipart([msg, msg_info], flags, copy=copy, track=track)

def recv_from_prev_raw(src):
    msg, msg_info = src.recv_multipart()
    return msg, msg_info

def send_ndarray(dst, array, flags=0, copy=True, track=False):
    md = dict(dtype=str(array.dtype), shape=array.shape)
    msg_info = jsonapi.dumps(md)
    send_to_next_raw(array, msg_info, dst, flags=flags, copy=copy, track=track )

def recv_ndarray(src):
    msg, msg_info = recv_from_prev_raw(src)
    arr_info, arr_val = jsonapi.loads(msg_info), msg
    array = decode_ndarray(arr_val, arr_info)
    return array

def decode_ndarray(buffer, info):
    return np.frombuffer(memoryview(buffer), dtype=info['dtype']).reshape(info['shape'])

def send_object(dst, obj, flags=0, copy=True, track=False, protocol=-1, need_compress=0):
    if need_compress == 1:
        p = pickle.dumps(obj, protocol)
        z = zlib.compress(p)
    else:
        z = pickle.dumps(obj, protocol)
    obj_info = jsonapi.dumps(dict(protocol=protocol, compress=need_compress))
    send_to_next_raw(z, obj_info, dst, flags=flags, copy=copy, track=track )

def recv_object(src):
    msg, msg_info = recv_from_prev_raw(src)
    obj_info, obj_buffer = jsonapi.loads(msg_info), msg
    obj = decode_object(obj_buffer, obj_info)
    return obj

def decode_object(buffer, info):
    pickle_protocol = info['protocol']
    need_decompress = info['compress']
    if need_decompress == 1:
        obj_decompressed = zlib.decompress(buffer)
        obj = pickle.loads(obj_decompressed)
    else:
        obj = pickle.loads(buffer)
    return obj

def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of bytes

def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of str