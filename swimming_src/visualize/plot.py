# Visualize Reults with MatPlotlib etc...

import matplotlib.pyplot as plt


class Plot(object):
    def __init__(self):
        super().__init__()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 6))

    def plot2d(self):
        pass

    def plot3d(self):
        pass

    def animePlay(self):
        pass


if __name__ == "__main__":
    # Unit Test Here
    pass
