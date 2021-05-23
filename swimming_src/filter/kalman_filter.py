import numpy as np
from .basic_filter import Filter


class KF(object):
    def __init__(self):
        super().__init__()

        # Basically, We assume that robot moves in 3D place
        self._state = np.zeros((1, 3))
        self._covariance = np.eye(3)

    # Set Initial state / cov
    def set_initial_state(self, initial_x, initial_cov):
        self._state = initial_x
        self._covariance = initial_cov

    # Brand-New Sensor data incoming
    def measurement(self):
        pass

    # Update current state / cov
    def update(self):
        pass

    # Actual Sensor Fusion Magic happends in here
    def propagate(self):
        pass


if __name__ == "__main__":
    # Unit Test Here
    pass
