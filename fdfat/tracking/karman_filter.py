import numpy as np
from scipy.linalg import block_diag

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def create_filter(dt=1, R_std=0.1, Q_std=0.04):
    
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
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker