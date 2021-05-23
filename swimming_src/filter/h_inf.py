import numpy as np
from .basic_filter import Filter


class HInf(Filter):
    def __init__(self):
        super().__init__()

        # Basically, We assume that robot moves in 3D place
        self._state = np.zeros((1, 3))
        self._covariance = np.eye(3)

        self.GAMMA = 0.025

    def set_gamma(self, gamma):
        self.GAMMA = gamma

    # Set Initial state / cov
    def set_initial_state(self, initial_x, initial_cov):
        self._state = initial_x
        self._covariance = initial_cov

    # Brand-New Sensor data incoming
    def measurement(self):
        pass

    # Update current state / cov
    def update(self, z, Q, R):
        """
        TODO: Docstring Here
        """

        S = self._covariance
        L = np.eye(3)
        S_bar = L @ S @ L.T
        I = np.eye(3)
        print(f"gamma : {self.GAMMA}")

        H = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )  # Jacobian of observation function
        P_bar = (
            I
            - self.GAMMA * S_bar @ self._covariance
            + H.T @ np.linalg.inv(R) @ H @ self._covariance
        )

        K = self._covariance @ np.linalg.inv(P_bar) @ H.T @ np.linalg.inv(R)

        satisfaction = (
            np.linalg.inv(S) - self.GAMMA * S_bar + H.T @ np.linalg.inv(R) @ H
        )

        if np.linalg.norm(satisfaction) <= 0:
            print(f"not satisfied {np.linalg.norm(satisfaction)}")
        # nu  = z -H@self._state
        # Pinf = np.linalg.inv(I-(Q/self.GAMMA/self.GAMMA)+self._covariance@H.T@H)@Qbar;
        # Qbar = Pinf + Q
        # L = np.linalg.inv(I-(Q/self.GAMMA/self.GAMMA)@self._covariance + H.T@np.linalg.inv(Q)@H@self._covariance)
        # K = self._covariance @ L @H.T@np.linalg.inv(Q)
        # K = Pinf@H.T

        # update state
        x, y, theta = self._state
        z_ = np.array([x, y, theta])  # expected observation from the estimated state
        self._state = self._state + K @ (z - z_)

        # update covariance P
        self._covariance = self._covariance - K @ H @ self._covariance

    # Actual Sensor Fusion Magic happends in here
    def propagate(self, u, dt, R):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """

        # propagate state x
        x, y, theta = self._state
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = -r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = +r * np.cos(theta) - r * np.cos(theta + dtheta)

        self._state += np.array([dx, dy, dtheta])

        self._covariance = self._covariance + R


if __name__ == "__main__":
    # Unit Test Here
    pass
