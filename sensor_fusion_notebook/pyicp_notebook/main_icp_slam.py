import os
import sys
import csv
import copy
import time
import random
import argparse

import numpy as np
np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

import pykitti

sys.path.append('src')

from minisam import *
from kalman_filters import ExtendedKalmanFilter as EKF

from pyicp import *
from utils import *

# from ScanContextManager import *
# from PoseGraphManager import *
# from UtilsMisc import *
# import UtilsPointcloud as Ptutils
# import ICP as ICP

# params
parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=5000) # 5000 is enough for real time

parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper

parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)

parser.add_argument('--data_base_dir', type=str, 
                    default='/home/kimsooyoung/Documents/AI_KR/dataset')
parser.add_argument('--sequence_idx', type=str, default='06')

parser.add_argument('--save_gap', type=int, default=300)

args = parser.parse_args()

# dataset
dataset = pykitti.odometry(args.data_base_dir, args.sequence_idx)
num_frames = len(dataset)

# Pose Graph Manager (for back-end optimization) initialization
PGM = PoseGraphManager()
PGM.addPriorFactor()

# Result saver
save_dir = "result/" + args.sequence_idx
if not os.path.exists(save_dir): os.makedirs(save_dir)
ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3, 
                             save_gap=args.save_gap,
                             num_frames=num_frames,
                             seq_idx=args.sequence_idx,
                             save_dir=save_dir)

# Scan Context Manager (for loop detection) initialization
SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors], 
                                        num_candidates=args.num_candidates, 
                                        threshold=args.loop_threshold)

# for save the results as a video
fig_idx = 1
fig = plt.figure(fig_idx)
# writer = FFMpegWriter(fps=15)
# video_name = args.sequence_idx + "_" + str(args.num_icp_points) + ".mp4"
num_frames_to_skip_to_show = 5
num_frames_to_save = np.floor(num_frames/num_frames_to_skip_to_show)

# with writer.saving(fig, video_name, num_frames_to_save): # this video saving part is optional

# first_velo = dataset.get_velo(0)
# second_velo = dataset.get_velo(1)

# first_idx = 0
# second_idx = 1

# # random sample current information
# curr_scan_down_pts = random_sampling(first_velo, num_points=args.num_icp_points)
# curr_scan_down_pts_2 = random_sampling(second_velo, num_points=args.num_icp_points)

# save current node
# PGM.curr_node_idx = first_idx # make start with 0
# SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_down_pts)
# if(PGM.curr_node_idx == 0):
#     PGM.prev_node_idx = PGM.curr_node_idx
#     prev_scan_pts = copy.deepcopy(first_velo)
#     icp_initial = np.eye(4)
#     pass

# PGM.prev_node_idx = PGM.curr_node_idx
# prev_scan_pts = copy.deepcopy(second_velo)
# icp_initial = np.eye(4)

# PGM.curr_node_idx = second_idx # make start with 0
# SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_down_pts_2)

# # calc odometry
# prev_scan_down_pts = random_sampling(prev_scan_pts, num_points=args.num_icp_points)
# odom_transform, _, _ = icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=icp_initial, max_iterations=20)

# # update the current (moved) pose 
# PGM.curr_se3 = np.matmul(PGM.curr_se3, odom_transform)
# icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)


# # renewal the prev information 
# PGM.prev_node_idx = PGM.curr_node_idx
# prev_scan_pts = copy.deepcopy(second_velo)

# loop detection and optimize the graph 
if(PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0): 
    # 1/ loop detection 
    loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
    if(loop_idx == None): # NOT FOUND
        pass
    else:
        print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
        # 2-1/ add the loop factor 
        loop_scan_down_pts = SCM.getPtcloud(loop_idx)
        loop_transform, _, _ = icp(curr_scan_down_pts, loop_scan_down_pts, init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
        PGM.addLoopFactor(loop_transform, loop_idx)

        # 2-2/ graph optimization 
        PGM.optimizePoseGraph()

        # 2-2/ save optimized poses
        ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

# # save the ICP odometry pose result (no loop closure)
# ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
# if(first_idx % num_frames_to_skip_to_show == 0):
#     ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)


# @@@ MAIN @@@: data stream
# for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):
for for_idx, curr_scan_pts in tqdm(enumerate(dataset.velo), total=num_frames, mininterval=5.0):

    # random sample current information     
    curr_scan_down_pts = random_sampling(curr_scan_pts, num_points=args.num_icp_points)

    # save current node
    PGM.curr_node_idx = for_idx # make start with 0
    SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_down_pts)
    if(PGM.curr_node_idx == 0):
        PGM.prev_node_idx = PGM.curr_node_idx
        prev_scan_pts = copy.deepcopy(curr_scan_pts)
        icp_initial = np.eye(4)
        continue

    # calc odometry
    prev_scan_down_pts = random_sampling(prev_scan_pts, num_points=args.num_icp_points)
    odom_transform, _, _ = icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=icp_initial, max_iterations=20)

    # update the current (moved) pose 
    PGM.curr_se3 = np.matmul(PGM.curr_se3, odom_transform)
    icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)

    # add the odometry factor to the graph 
    PGM.addOdometryFactor(odom_transform)

    # renewal the prev information 
    PGM.prev_node_idx = PGM.curr_node_idx
    prev_scan_pts = copy.deepcopy(curr_scan_pts)

    # loop detection and optimize the graph 
    if(PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0): 
        # 1/ loop detection 
        loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
        if(loop_idx == None): # NOT FOUND
            pass
        else:
            print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
            # 2-1/ add the loop factor 
            loop_scan_down_pts = SCM.getPtcloud(loop_idx)
            loop_transform, _, _ = icp(curr_scan_down_pts, loop_scan_down_pts, init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
            PGM.addLoopFactor(loop_transform, loop_idx)

            # 2-2/ graph optimization 
            PGM.optimizePoseGraph()

            # 2-2/ save optimized poses
            ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

    # save the ICP odometry pose result (no loop closure)
    ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
    if(for_idx % num_frames_to_skip_to_show == 0):
        ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
        # writer.grab_frame()
