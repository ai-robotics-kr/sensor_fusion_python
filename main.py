import argparse
import numpy as np

from sensor_fusion import (
    EKF,
    Visualization,
    KittiDatasetMgmt,
    dataset_mgmt,
    normalize_angles,
)

# temporal func for Readability during Development
def setupDataset():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kitti_root_dir",
        type=str,
        default="D:\\2011_09_30_drive_0033_sync",
        help="Root directory for loading Kitti dataset ",
    )

    parser.add_argument(
        "--kitti_date",
        type=str,
        default="2011_09_30",
        help="Kitti dataset time information",
    )

    parser.add_argument(
        "--kitti_drive", type=str, default="0033", help="Kitti dataset drive number"
    )

    parser.add_argument('--plot_data', dest='plot_data', action='store_true')
    parser.add_argument('--no-plot_data', dest='plot_data', action='store_false')
    parser.set_defaults(plot_data=True)

    parser.add_argument(
        "--filter",
        type=str,
        default="EKF",
        choices=["KF", "EKF", "UKF", "PF", "HINF", "LIDAR-SLAM", "VISION-SLAM"],
        help="Filter which you want to use.[KF,EKF,UKF,PF,HINF,LIDAR-SLAM,VISION-SLAM]",
    )

    args = parser.parse_args()

    dataset_mgmt = KittiDatasetMgmt(
        args.kitti_root_dir, args.kitti_date, args.kitti_drive
    )
    return dataset_mgmt, args


def main():

    # Data Initialization Operations
    dataset_mgmt, args = setupDataset()

    # GPS Gaussian Noise Addition!!
    dataset_mgmt.addGaussianNoiseToGPS()

    # Plot Data, You can Turn off this with argparse option
    if args.plot_data is True:
        viz = Visualization(dataset_mgmt)
        viz.plotGPStrajactory()
        viz.plotXYZtrajactory()
        viz.plotGTvalue()
        viz.plotNoisyData()

    # Prepare Entry Point
    x = dataset_mgmt.getInitialStateVec2D()
    P = dataset_mgmt.getInitialCovMat2D()
    Q = dataset_mgmt.getInitialMeasErrMat2D()
    R = dataset_mgmt.getNoiseCov2D()
    ts = dataset_mgmt.getTimeStamp()

    noisy_forward_velocities = dataset_mgmt.getNoisyForwardVelocities()
    noisy_yaw_rates = dataset_mgmt.getNoisyYawRates()
    noisy_trajectory_xyz = dataset_mgmt.getNoisyTrajXYZ()

    data_len = len(ts)

    # array to store estimated 2d pose [x, y, theta]
    estimated_state = [x]
    # array to store estimated error variance of 2d pose
    estimated_var = [[P[0, 0], P[1, 1], P[2, 2]]]

    try:
        # initialize Kalman filter
        ekf = EKF(x, P)
        

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
            ekf.statePrediction(u, dt, R)

            # get measurement `z = [x, y] + noise`
            z = np.array(
                [noisy_trajectory_xyz[0, t_idx], noisy_trajectory_xyz[1, t_idx]]
            )

            # update!
            ekf.measurementUpdate(z, Q)

            # save results
            estimated_state.append(ekf.state)
            estimated_var.append([ekf.cov[0, 0], ekf.cov[1, 1], ekf.cov[2, 2]])

            t_last = t

        estimated_state = np.array(estimated_state)
        estimated_var = np.array(estimated_var)

        # Visualize estimated results
        if args.plot_data is True:
            viz.getEstimatedValues(estimated_state, estimated_var)
            viz.plotEstimatedTraj()
            viz.plotEstimated2DStates()

        pass
    except Exception as e:
        print(e)
    finally:
        print(f"Done...")


if __name__ == "__main__":
    main()
