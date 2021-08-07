import matplotlib.pyplot as plt
import numpy as np
import platform
import pykitti

# from geo_utils.geo_transforms import lla_to_enu
from .geo_utils.geo_transforms import lla_to_enu

OS_PLATFORM = platform.system()

# Matplotlib Figure Size depends on the OS Platform used.
FIG_SIZE = {"Linux": {}, "Darwin": {}, "Windows": {}}


class KittiDatasetMgmt(object):
    def __init__(self, kitti_root_dir, kitti_date, kitti_drive):
        super().__init__()

        self.dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

        self.timestamps = np.array(self.dataset.timestamps)

        # [longitude(deg), latitude(deg), altitude(meter)] x N
        self.gt_trajectory_lla = []
        self.gt_yaws = []  # [yaw_angle(rad),] x N
        self.gt_yaw_rates = []  # [vehicle_yaw_rate(rad/s),] x N
        self.gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N
        self.ts = []  #

        self.generateGroundTruthSets()

        # set the initial position to the origin
        self.origin = self.gt_trajectory_lla[:, 0]
        self.gt_trajectory_xyz = lla_to_enu(self.gt_trajectory_lla, self.origin)

        # noisy data
        self.noisy_trajectory_xyz = []
        self.noisy_forward_velocities = []
        self.noisy_yaw_rates = []

        # params for noise addition
        self.xy_obs_noise_std = (
            5.0  # standard deviation of observation noise of x and y in meter
        )
        self.yaw_rate_noise_std = 0.02  # standard deviation of yaw rate in rad/
        self.forward_velocity_noise_std = 0.3
        self.initial_yaw_std = np.pi

    def reloadDataset(self, kitti_root_dir, kitti_date, kitti_drive):
        """Reset Class Variables and with Reloading Datset"""
        self.dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

        self.gt_trajectory_lla = []
        self.gt_yaws = []
        self.gt_yaw_rates = []
        self.gt_forward_velocities = []
        self.ts = []

        self.noisy_trajectory_xyz = []
        self.noisy_forward_velocities = []
        self.noisy_yaw_rates = []

    def generateGroundTruthSets(self):
        """
        Extract vehicle GPS trajectory, yaw angle, yaw rate, and forward velocity from KITTI OXTS senser packets.
        These are treated as ground-truth trajectory in this experiment.
        """
        for oxts_data in self.dataset.oxts:
            packet = oxts_data.packet
            self.gt_trajectory_lla.append([packet.lon, packet.lat, packet.alt])
            self.gt_yaws.append(packet.yaw)
            self.gt_yaw_rates.append(packet.wz)
            self.gt_forward_velocities.append(packet.vf)

        self.gt_trajectory_lla = np.array(self.gt_trajectory_lla).T
        self.gt_yaws = np.array(self.gt_yaws)
        self.gt_yaw_rates = np.array(self.gt_yaw_rates)
        self.gt_forward_velocities = np.array(self.gt_forward_velocities)

        # time that starts from 0 for visualization
        timestamps = np.array(self.timestamps)
        elapsed = np.array(timestamps) - timestamps[0]
        self.ts = [t.total_seconds() for t in elapsed]

    def getDatasetShape(self):
        """TODO: shape for all data types??"""
        return self.gt_trajectory_lla.shape
    
    def getGPSTraj(self):
        return self.gt_trajectory_xyz

    def getNoisyTrajXYZ(self):
        return self.noisy_trajectory_xyz

    def getNoisyForwardVelocities(self):
        return self.noisy_forward_velocities

    def getNoisyYawRates(self):
        return self.noisy_yaw_rates

    def getTimeStamp(self):
        return self.ts

    def getGTYaws(self):
        return self.gt_yaws

    # x
    def getInitialStateVec2D(self):
        """
        Suppose initial 2d position [x, y] estimation are initialized with the first GPS observation.
        Since our vehicle has no sensor to measure yaw angle,
        yaw estimation is initialized randomly and its variance is initialized with some large value (e.g. pi).
        2D version has (3, 1) Shape
        """

        self.initial_yaw_std = np.pi
        initial_yaw = self.gt_yaws[0] + np.random.normal(0, self.initial_yaw_std)

        initial_state_vec = np.array(
            [
                self.noisy_trajectory_xyz[0, 0],
                self.noisy_trajectory_xyz[1, 0],
                initial_yaw,
            ]
        )

        return initial_state_vec

    def getInitialPoseMat3D(self):
        """
        retrieve initial pose matrix from dataset
        Use output of this at the first of filter loop for ground truth heading
        """
        initial_pose_mat = self.dataset.oxts[0].T_w_imu
        initial_pose_mat[:-1, 3] = self.gt_trajectory_xyz[:, 0]

        return initial_pose_mat

    # P
    def getInitialCovMat2D(self):
        """
        Prepare initial covariance Matrix P
        covariance for initial state estimation error (Sigma_0)
        2D version has (3, 3) Shape
        """

        initial_cov_mat = np.array(
            [
                [self.xy_obs_noise_std ** 2.0, 0.0, 0.0],
                [0.0, self.xy_obs_noise_std ** 2.0, 0.0],
                [0.0, 0.0, self.initial_yaw_std ** 2.0],
            ]
        )
        return initial_cov_mat

    def getInitialCovMat3D(self):
        """"""
        # return initial_pose_mat
        print("Not Yet Implemented")
        return None

    # Q
    def getInitialMeasErrMat2D(self):
        """
        Prepare measuerment error covariance Q
        2D version has (2, 2) Shape
        """
        meas_err_mat = np.array(
            [
                [self.xy_obs_noise_std ** 2.0, 0.0], 
                [0.0, self.xy_obs_noise_std ** 2.0],
            ]
        )

        return meas_err_mat

    def getInitialMeasErrMat3D(self):
        pass



    # R
    def getNoiseCov2D(self):
        """Return Noise Covariance Mat for Measurement
        """
        noise_cov_mat = np.array([
            [self.forward_velocity_noise_std ** 2., 0., 0.],
            [0., self.forward_velocity_noise_std ** 2., 0.],
            [0., 0., self.yaw_rate_noise_std ** 2.]
        ])
        return noise_cov_mat

    def addGaussianNoiseToGPS(
        self, gt_trajectory_xyz=None, gt_yaw_rates=None, gt_forward_velocities=None,
    ):

        if gt_trajectory_xyz is None:
            gt_trajectory_xyz = self.gt_trajectory_xyz
        if gt_yaw_rates is None:
            gt_yaw_rates = self.gt_yaw_rates
        if gt_forward_velocities is None:
            gt_forward_velocities = self.gt_forward_velocities

        N = len(self.ts)  # number of data point

        # Add XY noise to XYZ traj
        xy_obs_noise = np.random.normal(
            0.0, self.xy_obs_noise_std, (2, N)
        )  # generate gaussian noise
        self.noisy_trajectory_xyz = gt_trajectory_xyz.copy()
        self.noisy_trajectory_xyz[
            :2, :
        ] += xy_obs_noise  # add the noise to ground-truth positions

        # Add noise to yaw rates
        yaw_rate_noise = np.random.normal(
            0.0, self.yaw_rate_noise_std, (N,)
        )  # gen gaussian noise
        self.noisy_yaw_rates = gt_yaw_rates.copy()
        self.noisy_yaw_rates += (
            yaw_rate_noise  # add the noise to ground-truth positions
        )

        # Add noise to forward velocities
        forward_velocity_noise = np.random.normal(
            0.0, self.forward_velocity_noise_std, (N,)
        )  # generate gaussian noise
        self.noisy_forward_velocities = gt_forward_velocities.copy()
        self.noisy_forward_velocities += (
            forward_velocity_noise  # add the noise to ground-truth positions
        )


if __name__ == "__main__":
    """Prepare Dataset as following example
    2011_09_30/
        ├── 2011_09_30_drive_0033_sync
        │   ├── image_00
        │   ├── image_01
        │   ├── image_02
        │   ├── image_03
        │   ├── oxts
        │   └── velodyne_points
        ├── calib_cam_to_cam.txt
        ├── calib_imu_to_velo.txt
        └── calib_velo_to_cam.txt
    """

    # Ubuntu Location
    # 20.04
    kitti_root_dir = "/home/swimming/Documents/Dataset" # Put your dataset location
    # 18.04
    # kitti_root_dir = "/home/kimsooyoung/Documents/AI_KR"  # Put your dataset location

    # Mac OS Location
    # kitti_root_dir = (
    #     "/Users/swimm_kim/Documents/Dataset/2011_09_30"  # Put your dataset location
    # )

    kitti_date = "2011_09_30"
    kitti_drive = "0033"

    test_mgmt = KittiDatasetMgmt(kitti_root_dir, kitti_date, kitti_drive)

    # test_mgmt.plotGPStrajactory()
    # test_mgmt.plotXYZtrajactory()
    # test_mgmt.plotGTvalue()

    # Noise Addition!!
    test_mgmt.addGaussianNoiseToGPS()
    # test_mgmt.plotNoisytrajactory()

    # Initial Vec/Mat Test
    x = test_mgmt.getInitialStateVec2D()
    P = test_mgmt.getInitialCovMat2D()
    Q = test_mgmt.getInitialMeasErrMat2D()
    R = test_mgmt.getNoiseCov2D()

    print(f"{x} \n {P} \n {Q} \n {R} \n")
    # Unit Test Here
    pass
