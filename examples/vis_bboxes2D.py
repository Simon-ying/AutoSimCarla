"""
Visualizing all bounding boxes in a frame. (In camera images)
"""
import cv2
import yaml
import os
import sys
import logging
from tqdm import tqdm

code_base = "."
sys.path.append(code_base)
from data_descriptor import KittiDescriptor, LidarDescriptor
     
# log_level = logging.DEBUG
log_level = logging.INFO
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

with open("configs.yaml") as f:
    config = yaml.safe_load(f)
data_root = config["SAVE_CONFIG"]["ROOT_PATH"]

img_dir = os.path.join(data_root, "image/{0:06}.png")
lidar_dir = os.path.join(data_root, "velodyne/{0:06}.bin")
lable_dir = os.path.join(data_root, "kitti_label/{0:06}.txt")
full_lable_dir = os.path.join(data_root, "all_label/{0:06}.txt")
lidar_bbox_dir = os.path.join(data_root, "lidar_bbox/{0:06}.txt")
calib_dir = os.path.join(data_root, "calib/{0:06}.txt")

num_frames = len(os.listdir(os.path.join(data_root, "velodyne")))
index = 100
print(f"PROCESSING {num_frames} frames.")
print(f"VISUALIZE frame {index}.")

# ALL RESULT FILES
img_file = img_dir.format(index)
lidar_file = lidar_dir.format(index)
lable_file = lable_dir.format(index)
full_lable_file = full_lable_dir.format(index)
calib_file = calib_dir.format(index)
lidar_bbox_file = lidar_bbox_dir.format(index)

objects_text = []
with open(lable_file, 'r') as f:
    lines = f.readlines()
    for l in lines:
        objects_text.append(l.strip("\n"))
        logging.debug(l.strip("\n"))

kitti_points = []
for obj in objects_text:
    temp = KittiDescriptor()
    temp.load_from_str(obj)
    kitti_points.append(temp)
    logging.debug(temp)

img = cv2.imread(img_file)
for obj in kitti_points:
    img = cv2.rectangle(img, (obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3]), (0, 0, 255), 1)

cv2.imshow(f"frame {index}", img)
cv2.waitKey(0)
cv2.destroyWindow(f"frame {index}")