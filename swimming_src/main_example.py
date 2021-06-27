import numpy as np

from dataset_mgmt.geo_utils import normalize_angles
from dataset_mgmt import KittiDatasetMgmt
from visualize import ResultVizPlotter
from filter import EKF

kitti_root_dir = "/Users/swimm_kim/Documents/Dataset"
# kitti_root_dir = "/home/swimming/Documents/Dataset"
kitti_date = "2011_09_30"
kitti_drive = "0033"

PLOT_DATA = False

if __name__ == "__main__":

    dataset_mgmt = KittiDatasetMgmt(kitti_root_dir, kitti_date, kitti_drive)

    # plot your dataset
    if PLOT_DATA is True:
        dataset_mgmt.plotGPStrajactory()
        dataset_mgmt.plotXYZtrajactory()
        dataset_mgmt.plotGTvalue()

    # Noise Addition!!
    dataset_mgmt.addGaussianNoiseToGPS()

    # Plot Noise-Added Data
    if PLOT_DATA is True:
        dataset_mgmt.plotNoisytrajactory()

    # Prepare Entry Point
    # P : covariance for initial state estimation error (Sigma_0)
    # Q : measuerment error covariance
    # R : state transition noise covariance
    x = dataset_mgmt.getInitialStateVec2D()
    P = dataset_mgmt.getInitialCovMat2D()
    Q = dataset_mgmt.getInitialMeasErrMat2D()
    R = dataset_mgmt.getNoiseCov2D()
    ts = dataset_mgmt.getTimeStamp()
    gt_traj = dataset_mgmt.getGPSTraj()
    gt_yaws = dataset_mgmt.getGTYaws()

    noisy_forward_velocities = dataset_mgmt.getNoisyForwardVelocities()
    noisy_yaw_rates = dataset_mgmt.getNoisyYawRates()
    noisy_trajectory_xyz = dataset_mgmt.getNoisyTrajXYZ()

    # Dataset length for traversing
    data_len = len(ts)

    try:
        # initialize Kalman filter
        ekf = EKF(x, P)

        # array to store estimated 2d pose [x, y, theta]
        mu_x = [
            x[0],
        ]
        mu_y = [
            x[1],
        ]
        mu_theta = [
            x[2],
        ]

        # array to store estimated error variance of 2d pose
        var_x = [
            P[0, 0],
        ]
        var_y = [
            P[1, 1],
        ]
        var_theta = [
            P[2, 2],
        ]

        t_last = 0.0

        for t_idx in range(1, data_len):

            t = ts[t_idx]
            dt = t - t_last

            # get control input `u = [v, omega] + noise`
            u = np.array([noisy_forward_velocities[t_idx], noisy_yaw_rates[t_idx]])

            # because velocity and yaw rate are multiplied with `dt` in state transition function,
            # its noise covariance must be multiplied with `dt**2.`
            R = R * (dt ** 2.0)

            # propagate!
            ekf.propagate(u, dt, R)

            # get measurement `z = [x, y] + noise`
            z = np.array(
                [noisy_trajectory_xyz[0, t_idx], noisy_trajectory_xyz[1, t_idx]]
            )

            # update!
            ekf.measurementUpdate(z, Q)

            # save estimated state to analyze later
            mu_x.append(ekf.state[0])
            mu_y.append(ekf.state[1])
            mu_theta.append(normalize_angles(ekf.state[2]))

            # save estimated variance to analyze later
            var_x.append(ekf.cov[0, 0])
            var_y.append(ekf.cov[1, 1])
            var_theta.append(ekf.cov[2, 2])

            t_last = t

        estimated_2D_pose = np.array([mu_x, mu_y, mu_theta])
        estimated_variance = np.array([var_x, var_y, var_theta])

        print("Filter Processing Done!!")

    except Exception as e:
        print(e)
    finally:
        # Visualize estimated results
        result_viewer = ResultVizPlotter()
        result_viewer.plotTraj(gt_traj, noisy_trajectory_xyz, estimated_2D_pose)
        result_viewer.plotEstimated2DStates(
            ts, gt_traj, estimated_2D_pose, estimated_variance, gt_yaws
        )
        print(f"Done...")
