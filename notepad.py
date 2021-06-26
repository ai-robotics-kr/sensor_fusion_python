# https://github.com/motokimura/kalman_filter_witi_kitti/blob/master/demo.ipynb
import numpy as np


class ExtendedKalmanFilter:
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """

    def __init__(self, x, P):
        """
        Args:
            x (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        self.x = x  #  [3,]
        self.P = P  #  [3, 3]

    def update(self, z, Q):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): obsrervation for [x_, y_]^T
            Q (numpy.array): observation noise covariance
        """
        # compute Kalman gain
        H = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )  # Jacobian of observation function

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + Q)

        # update state x
        x, y, theta = self.x
        z_ = np.array([x, y])  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)

        # update covariance P
        self.P = self.P - K @ H @ self.P

    def propagate(self, u, dt, R):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """
        # propagate state x
        x, y, theta = self.x
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = -r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = +r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.x += np.array([dx, dy, dtheta])

        # propagate covariance P
        G = np.array(
            [
                [1.0, 0.0, -r * np.cos(theta) + r * np.cos(theta + dtheta)],
                [0.0, 1.0, -r * np.sin(theta) + r * np.sin(theta + dtheta)],
                [0.0, 0.0, 1.0],
            ]
        )  # Jacobian of state transition function

        self.P = G @ self.P @ G.T + R


R = np.array(
    [
        [forward_velocity_noise_std ** 2.0, 0.0, 0.0],
        [0.0, forward_velocity_noise_std ** 2.0, 0.0],
        [0.0, 0.0, yaw_rate_noise_std ** 2.0],
    ]
)




# initialize Kalman filter
kf = EKF(x, P)

# array to store estimated 2d pose [x, y, theta]
mu_x = [x[0],]
mu_y = [x[1],]
mu_theta = [x[2],]

# array to store estimated error variance of 2d pose
var_x = [P[0, 0],]
var_y = [P[1, 1],]
var_theta = [P[2, 2],]

t_last = 0.
for t_idx in range(1, N):
    t = ts[t_idx]
    dt = t - t_last
    
    # get control input `u = [v, omega] + noise`
    u = np.array([
        obs_forward_velocities[t_idx],
        obs_yaw_rates[t_idx]
    ])
    
    # because velocity and yaw rate are multiplied with `dt` in state transition function,
    # its noise covariance must be multiplied with `dt**2.`
    R_ = R * (dt ** 2.)
    
    # propagate!
    kf.propagate(u, dt, R)
    
    # get measurement `z = [x, y] + noise`
    z = np.array([
        obs_trajectory_xyz[0, t_idx],
        obs_trajectory_xyz[1, t_idx]
    ])
    
    # update!
    kf.update(z, Q)
    
    # save estimated state to analyze later
    mu_x.append(kf.x[0])
    mu_y.append(kf.x[1])
    mu_theta.append(normalize_angles(kf.x[2]))
    
    # save estimated variance to analyze later
    var_x.append(kf.P[0, 0])
    var_y.append(kf.P[1, 1])
    var_theta.append(kf.P[2, 2])
    
    t_last = t
    

mu_x = np.array(mu_x)
mu_y = np.array(mu_y)
mu_theta = np.array(mu_theta)

var_x = np.array(var_x)
var_y = np.array(var_y)
var_theta = np.array(var_theta)