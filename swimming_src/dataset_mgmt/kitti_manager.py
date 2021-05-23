import numpy as np
import pykitti


def normalize_angles(angles):
    """
    Args:
        angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
    Returns:
        numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
    """
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return angles


def initial_pose_mat(dataset, xyz_vector):
    pose_mat = dataset.oxts[0].T_w_imu
    pose_mat[:-1, 3] = xyz_vector

    return pose_mat


def generate_gt_sets(dataset):
    # [longitude(deg), latitude(deg), altitude(meter)] x N
    gt_trajectory_lla = []
    gt_yaws = []  # [yaw_angle(rad),] x N
    gt_yaw_rates = []  # [vehicle_yaw_rate(rad/s),] x N
    gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

    for oxts_data in dataset.oxts:
        packet = oxts_data.packet
        gt_trajectory_lla.append([packet.lon, packet.lat, packet.alt])
        gt_yaws.append(packet.yaw)
        gt_yaw_rates.append(packet.wz)
        gt_forward_velocities.append(packet.vf)

    gt_trajectory_lla = np.array(gt_trajectory_lla).T
    gt_yaws = np.array(gt_yaws)
    gt_yaw_rates = np.array(gt_yaw_rates)
    gt_forward_velocities = np.array(gt_forward_velocities)

    return gt_trajectory_lla, gt_yaws, gt_yaw_rates, gt_forward_velocities


# Load Dataset
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

    def getInitialPoseMat(self):
        pass

    def plotTraj(self):
        pass


if __name__ == "__main__":
    kitti_root_dir = "/home/kimsooyoung/Documents/AI_KR"
    kitti_date = "2011_09_30"
    kitti_drive = "0033"
    UtilDataset(kitti_root_dir, kitti_date, kitti_drive)
    # Unit Test Here
    pass
