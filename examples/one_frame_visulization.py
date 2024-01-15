import cv2

import os
import sys
import logging
import numpy as np
import struct
import open3d as o3d
from tqdm import tqdm
code_base = "/home/ying/CARLA_0.9.13/PythonAPI/examples/DataGenerator/"
sys.path.append(code_base)
from data_descriptor import KittiDescriptor, LidarDescriptor
from data_utils import _create_bb_points, eulerAngles2rotationMat, bbox_cam2velo

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def pointInBox(point, box):
	L01 = box[1, :] - box[0, :]
	L0p = point - box[0, :]
	L1p = point - box[1, :]
	D1 = L01.dot(L0p.transpose()) * L01.dot(L1p.transpose())
	if D1 > 0:
		return False
	L03 = box[3, :] - box[0, :]
	L3p = point - box[3, :]
	D2 = L03.dot(L0p.transpose()) * L03.dot(L3p.transpose())
	if D2 > 0:
		return False
	L04 = box[4, :] - box[0, :]
	L4p = point - box[4, :]
	D3 = L04.dot(L0p.transpose()) * L04.dot(L4p.transpose())
	if D3 > 0:
		return False
	return True

def search_bounding_boxes(pcd, pcd_tree, box, threshold):
    point = (box[0, :] + box[6, :]) / 2
    # point = o3d.utility.Vector3dVector(point)
    dis = np.linalg.norm(box[0, :] - box[6, :]) / 2
    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, dis)
    new_idx = []
    for i in idx:
        temp_point = np.array(pcd.points[i])
        if pointInBox(temp_point, box):
            new_idx.append(i)
    if len(new_idx) < threshold:
        return False
    else:
        np.asarray(pcd.colors)[new_idx, :] = [0, 1, 0]
        return True
     
# log_level = logging.DEBUG
log_level = logging.INFO
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

data_root = "/home/ying/CARLA_0.9.13/PythonAPI/examples/DataGenerator/data/06/"
img_dir = os.path.join(data_root, "image/{0:06}.png")
lidar_dir = os.path.join(data_root, "velodyne/{0:06}.bin")
lable_dir = os.path.join(data_root, "kitti_label/{0:06}.txt")
full_lable_dir = os.path.join(data_root, "all_label/{0:06}.txt")
lidar_bbox_dir = os.path.join(data_root, "lidar_bbox/{0:06}.txt")
calib_dir = os.path.join(data_root, "calib/{0:06}.txt")

save_dyna = os.path.join(data_root, "dynamics")
save_stat = os.path.join(data_root, "statics")
save_bbox = os.path.join(data_root, "bboxes")
os.makedirs(save_dyna, exist_ok=True)
os.makedirs(save_stat, exist_ok=True)
os.makedirs(save_bbox, exist_ok=True)

dynamic_dir = os.path.join(save_dyna, "{0:06}.npy")
static_dir = os.path.join(save_stat, "{0:06}.npy")
bbox_dir = os.path.join(save_bbox, "{0:06}.npy")

num_frames = len(os.listdir(os.path.join(data_root, "velodyne")))
print(f"PROCESSING {num_frames} frames.")

vis_ = o3d.visualization.Visualizer()
vis_.create_window()
render_options = vis_.get_render_option()
render_options.point_size = 1
render_options.background_color = np.array([0, 0, 0])
point_list = o3d.geometry.PointCloud()
line_list = []

index = 343

img_file = img_dir.format(index)
lidar_file = lidar_dir.format(index)
lable_file = lable_dir.format(index)
full_lable_file = full_lable_dir.format(index)
calib_file = calib_dir.format(index)
lidar_bbox_file = lidar_bbox_dir.format(index)

dynamic_file = dynamic_dir.format(index)
static_file = static_dir.format(index)
bbox_file = bbox_dir.format(index)

if os.path.exists(img_file):
    logging.debug(img_file)
if os.path.exists(lidar_file):
    logging.debug(lidar_file)
if os.path.exists(lable_file):
    logging.debug(lable_file)

objects_text = []
# with open(lidar_bbox_file, 'r') as f:
# with open(lable_file, 'r') as f:
with open(full_lable_file, 'r') as f:
    lines = f.readlines()
    for l in lines:
        objects_text.append(l.strip("\n"))
        logging.debug(l.strip("\n"))

kitti_points = []
for obj in objects_text:
    # temp = LidarDescriptor()
    temp = KittiDescriptor()
    temp.load_from_str(obj)
    kitti_points.append(temp)
    logging.debug(temp)

if False:
    img = cv2.imread(img_file)
    for obj in kitti_points:
        img = cv2.rectangle(img, (obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3]), (0, 0, 255), 1)

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyWindow("test")

with open(calib_file, 'r') as f:
    lines = f.readlines()
    for l in lines:
        key, val = l.strip("\n").split(":")
        logging.debug(l.strip("\n"))
        if key == "Tr_velo_to_cam":
            velo2cam = np.array(list((map(float, val.split(" ")[1:])))).reshape((3, 4))
            velo2cam = np.vstack([velo2cam, np.array([0, 0, 0, 1]).reshape(1, 4)])
            logging.debug(np.linalg.inv(velo2cam))

cam2velo = np.linalg.inv(velo2cam)

lidar_points = read_bin_velodyne(lidar_file)
moving_3dbox = []
line_sets = []
corners_3d_velo_sets = []

for obj in kitti_points:
    # Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, z1, z2 = obj.bbox
    # corners_3d = [[Ax, Ay, z1], [Bx, By, z1], [Cx, Cy, z1], [Dx, Dy, z1],
    #               [Ax, Ay, z2], [Bx, By, z2], [Cx, Cy, z2], [Dx, Dy, z2]]
    # corners_3d = np.array(corners_3d)
    corners_3d = bbox_cam2velo(obj, cam2velo)

    corners_3d_velo_sets.append(corners_3d)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_points)
pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for j in range(len(lidar_points))]))

pcd_tree = o3d.geometry.KDTreeFlann(pcd)

filterd_corners_3d_sets = []
for corners_3d_velo in corners_3d_velo_sets:
    flag = search_bounding_boxes(pcd, pcd_tree, corners_3d_velo, 30)
    if flag:
        filterd_corners_3d_sets.append(corners_3d_velo)

for corners_3d_velo in filterd_corners_3d_sets:
    line_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], 
                    [4, 5], [5, 6], [4, 7], [6, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7]])
    colors = np.array([[0, 1, 0] for j in range(len(line_box))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_3d_velo)
    line_set.lines = o3d.utility.Vector2iVector(line_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_sets.append(line_set)

dynamic_points = []
static_points = []

for idx, p in enumerate(pcd.points):
    if pcd.colors[idx][0] == 0:
        dynamic_points.append(np.array(pcd.points[idx]))
    else:
        static_points.append(np.array(pcd.points[idx]))

# if len(filterd_corners_3d_sets) == 0:
#     filterd_corners_3d_sets = np.array([])
#     dynamic_points = np.array([])
# else:
#     filterd_corners_3d_sets = np.stack(filterd_corners_3d_sets)
#     dynamic_points = np.stack(dynamic_points)
# static_points = np.stack(static_points)


# np.save(dynamic_file, dynamic_points)
# np.save(static_file, static_points)
# np.save(bbox_file, filterd_corners_3d_sets)

point_list.points = pcd.points
point_list.colors = pcd.colors

# Visualization
vis_.add_geometry(point_list)
for line_set in line_sets:
    vis_.add_geometry(line_set)

vis_.run()
vis_.destroy_window()     
# print(lidar_points.shape)