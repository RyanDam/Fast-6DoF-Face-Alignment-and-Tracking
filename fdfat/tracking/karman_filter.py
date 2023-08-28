import numpy as np
from scipy.linalg import block_diag

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))

def create_point_filter(point, dt=1, R_std=0.1, Q_std=0.04):
    tracker = KalmanFilter(dim_x=4, dim_z=2)

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[point[0], 0, point[1], 0]]).T
    tracker.P = np.eye(4) * 100.
    return tracker

def create_bbox_filter(bbox, dt=1):
    tracker = KalmanFilter(dim_x=7, dim_z=4) 
    tracker.F = np.array([[1,0,0,0,dt,0,0],
                          [0,1,0,0,0,dt,0],
                          [0,0,1,0,0,0,dt],
                          [0,0,0,1,0,0,0], 

                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
    
    tracker.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

    tracker.R[2:,2:] *= 10.
    tracker.P[4:,4:] *= 100. # give high uncertainty to the unobservable initial velocities
    tracker.P *= 10.
    tracker.Q[-1,-1] *= 0.01
    tracker.Q[4:,4:] *= 0.01

    tracker.x[:4] = convert_bbox_to_z(bbox)

    return tracker