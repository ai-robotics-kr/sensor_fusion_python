import numpy as np


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
    pose_mat[0][3] = xyz_vector[0]
    pose_mat[1][3] = xyz_vector[1]
    pose_mat[2][3] = xyz_vector[2]
    
    return pose_mat

def generate_gt_sets(dataset):
    gt_trajectory_lla = []      # [longitude(deg), latitude(deg), altitude(meter)] x N
    gt_yaws = []                # [yaw_angle(rad),] x N
    gt_yaw_rates= []            # [vehicle_yaw_rate(rad/s),] x N
    gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

    for oxts_data in dataset.oxts:
        packet = oxts_data.packet
        gt_trajectory_lla.append([
            packet.lon,
            packet.lat,
            packet.alt
        ])
        gt_yaws.append(packet.yaw)
        gt_yaw_rates.append(packet.wz)
        gt_forward_velocities.append(packet.vf)

    gt_trajectory_lla = np.array(gt_trajectory_lla).T
    gt_yaws = np.array(gt_yaws)
    gt_yaw_rates = np.array(gt_yaw_rates)
    gt_forward_velocities = np.array(gt_forward_velocities)

    return gt_trajectory_lla, gt_yaws, gt_yaw_rates, gt_forward_velocities