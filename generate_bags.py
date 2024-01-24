import numpy as np
import os
import open3d as o3d
from tqdm import tqdm
import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import cloud_msgs.msg as cloud_msgs
from geometry_msgs.msg import Point
import nav_msgs.msg as nav_msgs
import tf.transformations as trans
import rosbag
import argparse
import shutil
from scipy.spatial.transform import Rotation as R

def point_cloud(points, parent_frame, timestamp):
    """ Creates a point cloud message.
    Args:
        points: Nx4 array of xyzI positions (m)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(['x','y','z','intensity'])]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.from_sec(timestamp))

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4),
        row_step=(itemsize * 4 * points.shape[0]),
        data=data
    )

def poseMsg(pose, parent_frame, child_frame, timestamp):
    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.from_sec(timestamp))
    odometry = nav_msgs.Odometry()
    odometry.header = header
    odometry.child_frame_id = child_frame

    pose = np.array(pose).reshape(-1, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    rotation = R.from_matrix(rotation).as_quat()
    odometry.pose.pose.orientation.x = rotation[0]
    odometry.pose.pose.orientation.y = rotation[1]
    odometry.pose.pose.orientation.z = rotation[2]
    odometry.pose.pose.orientation.w = rotation[3]

    odometry.pose.pose.position.x = translation[0]
    odometry.pose.pose.position.y = translation[1]
    odometry.pose.pose.position.z = translation[2]

    return odometry

def publishPose(pose, parent_frame, child_frame, timestamp, bag):
    odometry = poseMsg(pose, parent_frame, child_frame, timestamp)
    bag.write("/poses", odometry, rospy.Time.from_sec(timestamp))

def publishCloud(staticCloud, dynamicCloud, parent_frame, timestamp, bag):

    pubDynamicCloud = point_cloud(dynamicCloud, parent_frame, timestamp)
    pubStaticCloud = point_cloud(staticCloud, parent_frame, timestamp)
    bag.write("/dynamic_points", pubDynamicCloud, rospy.Time.from_sec(timestamp))
    bag.write("/velodyne_points", pubStaticCloud, rospy.Time.from_sec(timestamp))

def publishTrackBox(bounding_boxes, parent_frame, timestamp, bag):
    """
    bounding_boxes[i, 0: 3] = A[i]
    bounding_boxes[i, 6: 9] = B[i]
    ...
    matching: 0-A, 2-B, 6-C, 4-D, 1-A', 3-B', 7-C', 5-D' #ignore
    matching: 0-A, 1-B, 3-C, 4-D, 5-A', 6-B', 7-C', 8-D'
    """
    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.from_sec(timestamp))
    box_num = bounding_boxes.shape[0]
    A = []
    B = []
    C = []
    D = []
    zmin = []
    zmax = []
    for i in range(bounding_boxes.shape[0]):
        zmin.append(bounding_boxes[i, 4, 2])
        zmax.append(bounding_boxes[i, 0, 2])
        A.append(Point(bounding_boxes[i, 0, 0], bounding_boxes[i, 0, 1], bounding_boxes[i, 0, 2]))
        B.append(Point(bounding_boxes[i, 1, 0], bounding_boxes[i, 1, 1], bounding_boxes[i, 1, 2]))
        C.append(Point(bounding_boxes[i, 2, 0], bounding_boxes[i, 2, 1], bounding_boxes[i, 2, 2]))
        D.append(Point(bounding_boxes[i, 3, 0], bounding_boxes[i, 3, 1], bounding_boxes[i, 3, 2]))

    # print(bounding_boxes.shape)
    bboxes = cloud_msgs.trackbox()
    bboxes.header = header
    bboxes.box_num = box_num
    bboxes.A = A
    bboxes.B = B
    bboxes.C = C
    bboxes.D = D
    bboxes.zmin = zmin
    bboxes.zmax = zmax

    bag.write("/bounding_boxes", bboxes, rospy.Time.from_sec(timestamp))

argparser = argparse.ArgumentParser(
    description='Create bag files')
argparser.add_argument(
	'--seq',
	default='0000',
	type=str)

args = argparser.parse_args()

data_dir = args.seq
dynamics_path = os.path.join(data_dir, "dynamics")
statics_path = os.path.join(data_dir, "statics")
bboxes_path = os.path.join(data_dir, "bboxes")
poses_path = os.path.join(data_dir, "poses.txt")

data_list = os.listdir(dynamics_path)
frames = [int(i.split(".")[0]) for i in data_list]
min_frame = min(frames)
max_frame = max(frames)

times = np.linspace(1., 1.+(max_frame-min_frame)*0.1, (max_frame-min_frame+1))

with open(poses_path, "r") as f:
    lines = f.readlines()
    poses = [np.array(pose.strip().split(), dtype=float).reshape(-1, 4) for pose in lines]

poses = [np.r_[pose, np.array([[0, 0, 0, 1]])] for pose in poses]
pose0 = poses[0]
pose0_inv = np.linalg.inv(pose0)
poses = [pose0_inv.dot(pose) for pose in poses]


bag_path = os.path.join(data_dir, "test.bag")
bag = rosbag.Bag(bag_path, 'w')

bag_path_all = os.path.join(data_dir, "test_all.bag")
bag_all = rosbag.Bag(bag_path_all, 'w')

for i in tqdm(range(min_frame, max_frame+1), total=max_frame-min_frame+1):
    test_frame = "{:0>6}".format(i)

    dynamic_points_path = os.path.join(dynamics_path, f"{test_frame}.npy")
    static_points_path = os.path.join(statics_path, f"{test_frame}.npy")
    bounding_boxes_path = os.path.join(bboxes_path, f"{test_frame}.npy")

    dynamic_points = np.load(dynamic_points_path)
    static_points = np.load(static_points_path)
    bounding_boxes = np.load(bounding_boxes_path)

    
    if dynamic_points.shape[0] == 0:
        pass
    else:
        dynamic_points = np.hstack([dynamic_points, np.ones((dynamic_points.shape[0], 1))])
    static_points = np.hstack([static_points, np.ones((static_points.shape[0], 1))])
    if dynamic_points.shape[0] == 0:
        all_points = static_points
    else:
        all_points = np.vstack([dynamic_points, static_points])

    timestamp = times[i-min_frame]
    pos_i = i - min_frame
    pose = poses[pos_i]
    publishPose(pose, "base_link_init", "base_link", timestamp, bag)

    publishCloud(static_points, dynamic_points, "base_link", timestamp, bag)
    publishTrackBox(bounding_boxes, "base_link", timestamp, bag)

    publishCloud(all_points, dynamic_points, "base_link", timestamp, bag_all)
    publishTrackBox(bounding_boxes, "base_link", timestamp, bag_all)


bag.close()
bag_all.close()

# shutil.rmtree(dynamics_path)
# shutil.rmtree(statics_path)
# shutil.rmtree(bboxes_path)