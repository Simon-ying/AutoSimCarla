import cv2

import os
import sys
import logging
import numpy as np
import struct
import open3d as o3d
from tqdm import tqdm
"""
transform all lidar points to world coordinate
"""

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def transform(pointcloud, transform_matrix):
    new_pointcloud = []
    for i in range(len(pointcloud)):
        point = pointcloud[i]
        point = point.reshape(3, 1)
        point = transform_matrix[:3, :3] @ point + transform_matrix[:3, 3][:, np.newaxis]
        point = point.ravel()
        new_pointcloud.append(point)
    return new_pointcloud

data_dir = "data/00"
pose_path = os.path.join(data_dir, "poses.txt")
lidar_list = os.listdir(data_dir + "/velodyne")

lidar_list = [os.path.join(data_dir + "/velodyne", i) for i in lidar_list]
lidar_list = sorted(lidar_list)
with open(pose_path, "r") as f:
    lines = f.readlines()
    poses = [np.array(pose.strip().split(), dtype=float).reshape(-1, 4) for pose in lines]

poses = [np.r_[pose, np.array([[0, 0, 0, 1]])] for pose in poses]
pose0 = poses[0]
pose0_inv = np.linalg.inv(pose0)
poses = [pose0_inv.dot(pose) for pose in poses]

vis_ = o3d.visualization.Visualizer()
vis_.create_window()
render_options = vis_.get_render_option()
render_options.point_size = 1
render_options.background_color = np.array([0, 0, 0])
point_list = o3d.geometry.PointCloud()

all_point = []
for ind in tqdm(range(0, len(lidar_list))):
    lidar_points = read_bin_velodyne(lidar_list[ind])
    tr_lidar = transform(lidar_points, poses[ind])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tr_lidar)
    pcd = pcd.voxel_down_sample(voxel_size=1.0)
    vis_.add_geometry(pcd)

vis_.run()
vis_.destroy_window()  