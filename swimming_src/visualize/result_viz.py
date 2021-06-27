# Visualize Reults with MatPlotlib etc...
import matplotlib.pyplot as plt
import numpy as np


class ResultVizPlotter(object):
    def __init__(self):
        super().__init__()

    def plotTraj(self, groud_truth, noisy_traj, estimated_traj):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        gt_xs, gt_ys, _ = groud_truth
        ax.plot(gt_xs, gt_ys, lw=2, label='ground-truth trajectory')

        noisy_xs, noisy_ys, _ = noisy_traj
        ax.plot(noisy_xs, noisy_ys, lw=0, marker='.', markersize=4, alpha=1., label='observed trajectory')

        est_xs, ext_ys, _ = estimated_traj
        ax.plot(est_xs, ext_ys, lw=2, label='estimated trajectory', color='r')

        plt.title("Trajactory Comparison")
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()

        plt.show()

    def plotEstimated2DStates(self, ts, gt_trajectory_xyz, estimated_traj, estimated_var):
        fig, ax = plt.subplots(2, 3, figsize=(14, 6))

        ax[0, 0].plot(ts, gt_trajectory_xyz[0], lw=2, label='ground-truth')
        ax[0, 0].plot(ts, estimated_traj[0], lw=1, label='estimated', color='r')
        ax[0, 0].set_xlabel("time elapsed [sec]")
        ax[0, 0].set_ylabel("X [m]")
        ax[0, 0].legend()
        
        plt.show()

    def plot3d(self):
        pass

    def animePlay(self):
        pass


if __name__ == "__main__":
    # Unit Test Here
    pass

