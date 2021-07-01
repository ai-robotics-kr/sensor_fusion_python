import numpy as np
from .basic_filter import Filter


class KF(object):
    def __init__(self):
        super().__init__()

        # Basically, We assume that robot moves in 2D place
        # state will be [x, y, theta]
        self.state = np.zeros((1, 3))
        self.covariance = np.eye(3)

    def setInitialState(self, initial_state, initial_cov):
        """Set Initial State & Covariance"""
        self.state = initial_state
        self.covariance = initial_cov

    def measurement(self):
        """Brand-New Sensor data incoming"""
        pass

    def update(self):
        """Update current State & Covariance"""
        pass

    def propagate(self):
        """Filter Applied Part, Actual Sensor Fusion Magic happends in here """
        pass


if __name__ == "__main__":
    # Unit Test Here
    pass
