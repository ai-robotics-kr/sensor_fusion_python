import matplotlib.pyplot as plt
import numpy as np
import pykitti

from geo_transforms import lla_to_enu


def normalize_angles(angles):
    """
    Args:
        angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
    Returns:
        numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
    """
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return angles

class KittiDatasetMgmt(object):
    def __init__(self, kitti_root_dir, kitti_date, kitti_drive):
        super().__init__()

        self.dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

        # Ground Truth Values
        self.gt_forward_velocities = []
        self.gt_trajectory_lla = []
        self.gt_yaw_rates = []
        self.gt_yaws = []

        self.generateGroundTruthSets()

        # set the initial position to the origin
        self.origin = self.gt_trajectory_lla[:, 0]
        self.gt_trajectory_xyz = lla_to_enu(self.gt_trajectory_lla, self.origin)

    def reloadDataset(self, kitti_root_dir, kitti_date, kitti_drive):
        self.dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

    def generateGroundTruthSets(self):
        # [longitude(deg), latitude(deg), altitude(meter)] x N
        gt_trajectory_lla = []
        gt_yaws = []  # [yaw_angle(rad),] x N
        gt_yaw_rates = []  # [vehicle_yaw_rate(rad/s),] x N
        gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

        for oxts_data in self.dataset.oxts:
            packet = oxts_data.packet
            gt_trajectory_lla.append([packet.lon, packet.lat, packet.alt])
            gt_yaws.append(packet.yaw)
            gt_yaw_rates.append(packet.wz)
            gt_forward_velocities.append(packet.vf)

        gt_trajectory_lla = np.array(gt_trajectory_lla).T
        gt_yaws = np.array(gt_yaws)
        gt_yaw_rates = np.array(gt_yaw_rates)
        gt_forward_velocities = np.array(gt_forward_velocities)

    def getDatasetShape(self):
        """TODO: shape for all data types??
        """
        return self.gt_trajectory_lla.shape

    def getInitialPoseMat(self):
        """retrieve initial pose matrix from dataset
        Use output of this at the first of filter loop for ground truth heading
        """
        initial_pose_mat = self.dataset.oxts[0].T_w_imu
        initial_pose_mat[:-1, 3] = self.gt_trajectory_xyz[:, 0]

        return initial_pose_mat

    def plotGPStrajactory(self):
        lons, lats, _ = self.gt_trajectory_lla

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(lons, lats)
        ax.set_xlabel('longitude [deg]')
        ax.set_ylabel('latitude [deg]')
        ax.grid()
        ax.show()

if __name__ == "__main__":
    kitti_root_dir = "/home/kimsooyoung/Documents/AI_KR" # Put your dataset location
    kitti_date = "2011_09_30"
    kitti_drive = "0033"
    
    test_mgmt = KittiDatasetMgmt(kitti_root_dir, kitti_date, kitti_drive)
    
    test_mgmt.plotGPStrajactory()

    # Unit Test Here
    pass

