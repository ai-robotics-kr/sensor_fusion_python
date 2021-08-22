# Visualize Reults with MatPlotlib etc...
import matplotlib.pyplot as plt
import numpy as np

class Visualization(object):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def plotGPStrajactory(self):
        """
        run after generateGroundTruthSets, plot trajactory with GPS lng/lat data
        """
        lons, lats, _ = self.dataset.gt_trajectory_lla

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(lons, lats)
        plt.title("GPS trajactory")
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.grid()
        plt.show()


    def plotXYZtrajactory(self):
        """
        lla_to_enu converts lng/lat/alt to enu form
        plot converted XYZ trajectory
        """
        xs, ys, _ = self.dataset.gt_trajectory_xyz

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(xs, ys)
        plt.title("XYZ trajactory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid()
        plt.show()


    def plotGTvalue(self):
        """
        plot Ground-Truth Yaw angles / Yaw rates /Forward velocity
        ts required for plotlib x-axis
        """
        fig, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1, 1]}, figsize=(10, 14))

        ax[0].plot(self.dataset.ts, self.dataset.gt_yaws)
        ax[0].title.set_text("Ground-Truth yaw angles")
        ax[0].set_xlabel("time elapsed [sec]")
        ax[0].set_ylabel("ground-truth yaw angle [rad]")
        # ax[0].legend()

        ax[1].plot(self.dataset.ts, self.dataset.gt_yaw_rates)
        ax[1].title.set_text("Yaw Rates")
        ax[1].set_xlabel("time elapsed [sec]")
        ax[1].set_ylabel("ground-truth yaw rate [rad/s]")
        # ax[1].legend()

        ax[2].plot(self.dataset.ts, self.dataset.gt_forward_velocities)
        ax[2].title.set_text("Forward Velocitis")
        ax[2].set_xlabel("time elapsed [sec]")
        ax[2].set_ylabel("ground-truth forward velocity [m/s]")
        # ax[2].legend()

        plt.show()


    # TODO No handles with labels found to put in legend.
    def plotNoisyData(self):
        """
        After addGaussianNoiseToGPS, plot 3 types of noisy/ground-truth data at same plot

        1. GT/Noisy XYZ Traj
        2. GT/Noisy Yaw Rates
        3. GT/Noisy Forward Velocities
        """
        fig, ax = plt.subplots(
            3, 1, gridspec_kw={"height_ratios": [2, 1, 1]}, figsize=(12, 20)
        )

        # Plot1 GT/Noisy Traj
        gt_xs, gt_ys, _ = self.dataset.gt_trajectory_xyz
        noisy_xs, noisy_ys, _ = self.dataset.noisy_trajectory_xyz
        ax[0].plot(gt_xs, gt_ys, lw=2, label="ground-truth trajectory")
        ax[0].plot(
            noisy_xs,
            noisy_ys,
            lw=0,
            marker=".",
            markersize=5,
            alpha=0.4,
            label="noisy trajectory",
        )
        ax[0].set_xlabel("X [m]")
        ax[0].set_ylabel("Y [m]")
        ax[0].legend()
        ax[0].grid()

        # Plot2 GT/Noisy Yaw Rates
        ax[1].plot(self.dataset.ts, self.dataset.gt_yaw_rates, lw=1, label="ground-truth")
        ax[1].plot(
            self.dataset.ts, self.dataset.noisy_yaw_rates, lw=0, marker=".", alpha=0.4, label="noisy"
        )
        ax[1].set_xlabel("time elapsed [sec]")
        ax[1].set_ylabel("yaw rate [rad/s]")
        ax[1].legend()

        # Plot3 GT/Noisy Forward Velocities
        ax[2].plot(self.dataset.ts, self.dataset.gt_forward_velocities, lw=1, label="ground-truth")
        ax[2].plot(self.dataset.ts,
        self.dataset.noisy_forward_velocities,
        lw=0,
        marker=".",
        alpha=0.4,
        label="noisy",
        )
        ax[2].set_xlabel("time elapsed [sec]")
        ax[2].set_ylabel("forward velocity [m/s]")
        ax[2].legend()
        plt.show()

    def plotTraj(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        gt_xs, gt_ys, _ = self.dataset.groud_truth
        ax.plot(gt_xs, gt_ys, lw=2, label="ground-truth trajectory")

        noisy_xs, noisy_ys, _ = self.dataset.noisy_traj
        ax.plot(
            noisy_xs,
            noisy_ys,
            lw=0,
            marker=".",
            markersize=4,
            alpha=1.0,
            label="observed trajectory",
        )

        est_xs, ext_ys, _ = self.dataset.estimated_traj
        ax.plot(est_xs, ext_ys, lw=2, label="estimated trajectory", color="r")

        plt.title("Trajactory Comparison")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.legend()
        ax.grid()

        plt.show()

    def plotEstimated2DStates(self):
        fig, ax = plt.subplots(2, 3, figsize=(14, 6))

        # Analyze estimation error of X
        ax[0, 0].plot(self.dataset.ts, self.dataset.gt_trajectory_xyz[0], lw=2, label="ground-truth")
        ax[0, 0].plot(self.dataset.ts, self.dataset.estimated_traj[0], lw=1, label="estimated", color="r")
        ax[0, 0].set_xlabel("time elapsed [sec]")
        ax[0, 0].set_ylabel("X [m]")
        ax[0, 0].legend()

        ax[1, 0].plot(
            self.dataset.ts,
            self.dataset.estimated_traj[0] - self.dataset.gt_trajectory_xyz[0],
            lw=1.5,
            label="estimation error",
            )
        ax[1, 0].plot(
            self.dataset.ts,
            np.sqrt(self.dataset.estimated_var[0]),
            lw=1.5,
            label="estimated 1-sigma interval",
            color="darkorange",
            )
        ax[1, 0].plot(
            self.dataset.ts, -np.sqrt(self.dataset.estimated_var[0]), lw=1.5, label="", color="darkorange"
            )
        ax[1, 0].set_xlabel("time elapsed [sec]")
        ax[1, 0].set_ylabel("X estimation error [m]")
        ax[1, 0].legend()

        # Analyze estimation error of Y
        ax[0, 1].plot(self.dataset.ts, self.dataset.gt_trajectory_xyz[1], lw=2, label="ground-truth")
        ax[0, 1].plot(self.dataset.ts, self.dataset.estimated_traj[1], lw=1, label="estimated", color="r")
        ax[0, 1].set_xlabel("time elapsed [sec]")
        ax[0, 1].set_ylabel("Y [m]")
        ax[0, 1].legend()

        ax[1, 1].plot(
            self.dataset.ts,
            self.dataset.estimated_traj[1] - self.dataset.gt_trajectory_xyz[1],
            lw=1.5,
            label="estimation error",
            )
        ax[1, 1].plot(
            self.dataset.ts,
            np.sqrt(self.dataset.estimated_var[1]),
            lw=1.5,
            label="estimated 1-sigma interval",
            color="darkorange",
            )
        ax[1, 1].plot(
            self.dataset.ts, -np.sqrt(self.dataset.estimated_var[1]), lw=1.5, label="", color="darkorange"
            )
        ax[1, 1].set_xlabel("time elapsed [sec]")
        ax[1, 1].set_ylabel("Y estimation error [m]")
        ax[1, 1].legend()

        # Analyze estimation error of Theta
        ax[0, 2].plot(self.dataset.ts, self.dataset.gt_trajectory_xyz[2], lw=2, label="ground-truth")
        ax[0, 2].plot(self.dataset.ts, self.dataset.estimated_traj[2], lw=1, label="estimated", color="r")
        ax[0, 2].set_xlabel("time elapsed [sec]")
        ax[0, 2].set_ylabel("yaw angle [rad/s]")
        ax[0, 2].legend()

        ax[1, 2].plot(
            self.dataset.ts,
            normalize_angles(self.dataset.estimated_traj[2] - self.dataset.gt_yaws),
            lw=1.5,
            label="estimation error",
            )
        ax[1, 2].plot(
            self.dataset.ts,
            np.sqrt(self.dataset.estimated_var[2]),
            lw=1.5,
            label="estimated 1-sigma interval",
            color="darkorange",
            )
        ax[1, 2].plot(
            self.dataset.ts, -np.sqrt(self.dataset.estimated_var[2]), lw=1.5, label="", color="darkorange"
            )
        ax[1, 2].set_ylim(-0.5, 0.5)
        ax[1, 2].set_xlabel("time elapsed [sec]")
        ax[1, 2].set_ylabel("yaw estimation error [rad]")
        ax[1, 2].legend()

        plt.show()

    def normalize_angles(self):
        """
        Args:
        angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
        Returns:
            numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
        """
        norm_angles = (self.dataset.angles + np.pi) % (2 * np.pi) - np.pi
        return norm_angles

    def plot3d(self):
        pass

    def animePlay(self):
        pass

if __name__ == "__main__":
    # Unit Test Here
    pass
