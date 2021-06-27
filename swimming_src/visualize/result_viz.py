# Visualize Reults with MatPlotlib etc...
import matplotlib.pyplot as plt
import numpy as np


class ResultVizPlotter(object):
    def __init__(self):
        super().__init__()

    def plotTraj(self, groud_truth, noisy_traj, estimated_traj):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        gt_xs, gt_ys, _ = groud_truth
        ax.plot(gt_xs, gt_ys, lw=2, label="ground-truth trajectory")

        noisy_xs, noisy_ys, _ = noisy_traj
        ax.plot(
            noisy_xs,
            noisy_ys,
            lw=0,
            marker=".",
            markersize=4,
            alpha=1.0,
            label="observed trajectory",
        )

        est_xs, ext_ys, _ = estimated_traj
        ax.plot(est_xs, ext_ys, lw=2, label="estimated trajectory", color="r")

        plt.title("Trajactory Comparison")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.legend()
        ax.grid()

        plt.show()

    def plotEstimated2DStates(
        self, ts, gt_trajectory_xyz, estimated_traj, estimated_var, gt_yaws
    ):
        fig, ax = plt.subplots(2, 3, figsize=(14, 6))

        # Analyze estimation error of X
        ax[0, 0].plot(ts, gt_trajectory_xyz[0], lw=2, label="ground-truth")
        ax[0, 0].plot(ts, estimated_traj[0], lw=1, label="estimated", color="r")
        ax[0, 0].set_xlabel("time elapsed [sec]")
        ax[0, 0].set_ylabel("X [m]")
        ax[0, 0].legend()

        ax[1, 0].plot(
            ts,
            estimated_traj[0] - gt_trajectory_xyz[0],
            lw=1.5,
            label="estimation error",
        )
        ax[1, 0].plot(
            ts,
            np.sqrt(estimated_var[0]),
            lw=1.5,
            label="estimated 1-sigma interval",
            color="darkorange",
        )
        ax[1, 0].plot(
            ts, -np.sqrt(estimated_var[0]), lw=1.5, label="", color="darkorange"
        )
        ax[1, 0].set_xlabel("time elapsed [sec]")
        ax[1, 0].set_ylabel("X estimation error [m]")
        ax[1, 0].legend()

        # Analyze estimation error of Y
        ax[0, 1].plot(ts, gt_trajectory_xyz[1], lw=2, label="ground-truth")
        ax[0, 1].plot(ts, estimated_traj[1], lw=1, label="estimated", color="r")
        ax[0, 1].set_xlabel("time elapsed [sec]")
        ax[0, 1].set_ylabel("Y [m]")
        ax[0, 1].legend()

        ax[1, 1].plot(
            ts,
            estimated_traj[1] - gt_trajectory_xyz[1],
            lw=1.5,
            label="estimation error",
        )
        ax[1, 1].plot(
            ts,
            np.sqrt(estimated_var[1]),
            lw=1.5,
            label="estimated 1-sigma interval",
            color="darkorange",
        )
        ax[1, 1].plot(
            ts, -np.sqrt(estimated_var[1]), lw=1.5, label="", color="darkorange"
        )
        ax[1, 1].set_xlabel("time elapsed [sec]")
        ax[1, 1].set_ylabel("Y estimation error [m]")
        ax[1, 1].legend()

        # Analyze estimation error of Theta
        ax[0, 2].plot(ts, gt_trajectory_xyz[2], lw=2, label="ground-truth")
        ax[0, 2].plot(ts, estimated_traj[2], lw=1, label="estimated", color="r")
        ax[0, 2].set_xlabel("time elapsed [sec]")
        ax[0, 2].set_ylabel("yaw angle [rad/s]")
        ax[0, 2].legend()

        ax[1, 2].plot(
            ts,
            self.normalize_angles(estimated_traj[2] - gt_yaws),
            lw=1.5,
            label="estimation error",
        )
        ax[1, 2].plot(
            ts,
            np.sqrt(estimated_var[2]),
            lw=1.5,
            label="estimated 1-sigma interval",
            color="darkorange",
        )
        ax[1, 2].plot(
            ts, -np.sqrt(estimated_var[2]), lw=1.5, label="", color="darkorange"
        )
        ax[1, 2].set_ylim(-0.5, 0.5)
        ax[1, 2].set_xlabel("time elapsed [sec]")
        ax[1, 2].set_ylabel("yaw estimation error [rad]")
        ax[1, 2].legend()

        plt.show()

    def normalize_angles(self, angles):
        """
        Args:
            angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
        Returns:
            numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
        """
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        return angles

    def plot3d(self):
        pass

    def animePlay(self):
        pass


if __name__ == "__main__":
    # Unit Test Here
    pass
