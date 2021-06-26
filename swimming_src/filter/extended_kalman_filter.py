import numpy as np
from .basic_filter import Filter


class EKF(Filter):
    def __init__(self, x, P):
        super().__init__()

        # Basically, We assume that robot moves in 3D place
        self.state = x
        self.cov = P

    # Set Initial state / cov
    def setInitialState(self, initial_x, initial_cov):
        self.state = initial_x
        self.cov = initial_cov

    def measurementUpdate(self, z, Q):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): obsrervation for [x_, y_]^T
            Q (numpy.array): observation noise covariance
        """
        # compute Kalman gain
        H = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )  # Jacobian of observation function

        K = self.cov @ H.T @ np.linalg.inv(H @ self.cov @ H.T + Q)

        # update state x
        x, y, theta = self.state
        z_ = np.array([x, y])  # expected observation from the estimated state
        self.state = self.state + K @ (z - z_)

        # update covariance
        self.cov = self.cov - K @ H @ self.cov

    # Actual Sensor Fusion Magic happends in here
    def propagate(self, u, dt, R):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """
        # propagate state x
        x, y, theta = self.state
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = -r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = +r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.state += np.array([dx, dy, dtheta])

        # propagate covariance P
        G = np.array(
            [
                [1.0, 0.0, -r * np.cos(theta) + r * np.cos(theta + dtheta)],
                [0.0, 1.0, -r * np.sin(theta) + r * np.sin(theta + dtheta)],
                [0.0, 0.0, 1.0],
            ]
        )  # Jacobian of state transition function

        self.cov = G @ self.cov @ G.T + R


if __name__ == "__main__":
    # Unit Test Here
    pass
