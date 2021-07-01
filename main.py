import numpy as np

from sensor_fusion.dataset_mgmt.geo_utils import normalize_angles
from sensor_fusion.dataset_mgmt import KittiDatasetMgmt
from sensor_fusion.filter import EKF
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kitti_root_dir',
        type=str,
        default="/home/swimming/Documents/Dataset",
        help='Root directory for loading Kitti dataset '
    )
    parser.add_argument(
        '--kitti_date',
        type=str,
        default="2011_09_30",
        help='Kitti dataset time information')

    parser.add_argument(
        '--kitti_drive',
        type=str,
        default="0033",
        help='Kitti dataset drive number')

    parser.add_argument('--plot_data',
                        type=bool,
                        default=False,
                        help='Device used for model inference.')

    args = parser.parse_args()
    dataset_mgmt = KittiDatasetMgmt(args.kitti_root_dir, args.kitti_date, kitti_drive)

    # plot your dataset
    if args.plot_data is True:
        dataset_mgmt.plotGPStrajactory()
        dataset_mgmt.plotXYZtrajactory()
        dataset_mgmt.plotGTvalue()

    # Noise Addition!!
    dataset_mgmt.addGaussianNoiseToGPS()

    # Plot Noise-Added Data
    if args.plot_data is True:
        dataset_mgmt.plotNoisytrajactory()

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

    try:
        # initialize Kalman filter
        ekf = EKF(x, P)

        # array to store estimated 2d pose [x, y, theta]
        mu_x = [x[0], ]
        mu_y = [x[1], ]
        mu_theta = [x[2], ]

        # array to store estimated error variance of 2d pose
        var_x = [P[0, 0], ]
        var_y = [P[1, 1], ]
        var_theta = [P[2, 2], ]

        t_last = 0.0

        for t_idx in range(1, data_len):
            t = ts[t_idx]
            dt = t - t_last

            # get control input `u = [v, omega] + noise`
            u = np.array([
                noisy_forward_velocities[t_idx],
                noisy_yaw_rates[t_idx]
            ])

            # because velocity and yaw rate are multiplied with `dt` in state transition function,
            # its noise covariance must be multiplied with `dt**2.`
            R = R * (dt ** 2.)

            # propagate!
            ekf.propagate(u, dt, R)

            # get measurement `z = [x, y] + noise`
            z = np.array([
                noisy_trajectory_xyz[0, t_idx],
                noisy_trajectory_xyz[1, t_idx]
            ])

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

        mu_x = np.array(mu_x)
        mu_y = np.array(mu_y)
        mu_theta = np.array(mu_theta)

        var_x = np.array(var_x)
        var_y = np.array(var_y)
        var_theta = np.array(var_theta)

        print(mu_x.shape)
        print(mu_y.shape)
        print(mu_theta.shape)

        pass
    except Exception as e:
        print(e)
    finally:
        print(f"Done...")


if __name__ == "__main__":
    main()

