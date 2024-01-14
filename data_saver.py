from __init__ import *

from utils import config_to_trans
from export_utils import *

class DataSave:
    def __init__(self, cfg):
        self.cfg = cfg
        self.OUTPUT_FOLDER = None
        self.LIDAR_PATH = None
        self.KITTI_LABEL_PATH = None
        self.CARLA_LABEL_PATH = None
        self.ALL_LABEL_PATH = None
        self.IMAGE_PATH = None
        self.CALIBRATION_PATH = None
        self.LIDAR_BBOX_PATH = None
        self._generate_path(self.cfg["SAVE_CONFIG"]["ROOT_PATH"])
        self.captured_frame_no = self._current_captured_frame_num()


    def _generate_path(self,root_path):
        """ Generate files to store data"""
        self.OUTPUT_FOLDER = root_path
        folders = ['calib', 'image', 'kitti_label', 'carla_label', 'all_label', 'velodyne', 'lidar_bbox']

        if os.path.exists(self.OUTPUT_FOLDER):
            shutil.rmtree(self.OUTPUT_FOLDER)

        for folder in folders:
            directory = os.path.join(self.OUTPUT_FOLDER, folder)
            os.makedirs(directory)


        self.LIDAR_PATH = os.path.join(self.OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
        self.KITTI_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'kitti_label/{0:06}.txt')
        self.CARLA_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'carla_label/{0:06}.txt')
        self.ALL_LABEL_PATH = os.path.join(self.OUTPUT_FOLDER, 'all_label/{0:06}.txt')
        self.IMAGE_PATH = os.path.join(self.OUTPUT_FOLDER, 'image/{0:06}.png')
        self.CALIBRATION_PATH = os.path.join(self.OUTPUT_FOLDER, 'calib/{0:06}.txt')
        self.LIDAR_BBOX_PATH = os.path.join(self.OUTPUT_FOLDER, 'lidar_bbox/{0:06}.txt')


    def _current_captured_frame_num(self):
        """Get the frames existed in dataset"""
        label_path = os.path.join(self.OUTPUT_FOLDER, 'image/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.png')])
        print("There already exists {} frames in dataset".format(num_existing_data_files))
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                self.OUTPUT_FOLDER))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def save_training_files(self, lidar, rgb, depth, information):
        actors = information["actors"]
        agent = information["agent"]
        world_2_lidar = information["world_2_lidar"]
        extrinsic = information["extrinsic"]
        intrinsic = information["intrinsic"]
        
        lidar_fname = self.LIDAR_PATH.format(self.captured_frame_no)
        kitti_label_fname = self.KITTI_LABEL_PATH.format(self.captured_frame_no)
        carla_label_fname = self.CARLA_LABEL_PATH.format(self.captured_frame_no)
        all_label_fname = self.ALL_LABEL_PATH.format(self.captured_frame_no)
        img_fname = self.IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = self.CALIBRATION_PATH.format(self.captured_frame_no)
        lidar_bbox_frame = self.LIDAR_BBOX_PATH.format(self.captured_frame_no)


        camera_transform= config_to_trans(self.cfg["SENSOR_CONFIG"]["RGB"]["TRANSFORM"])
        lidar_transform = config_to_trans(self.cfg["SENSOR_CONFIG"]["LIDAR"]["TRANSFORM"])

        # save_ref_files(self.OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(img_fname, rgb)
        save_calibration_matrices([camera_transform, lidar_transform], calib_filename, intrinsic)
        save_lidar_data(lidar_fname, lidar)

        save_objects(kitti_label_fname, carla_label_fname, all_label_fname, lidar_bbox_frame, depth, agent, actors, intrinsic, extrinsic, world_2_lidar)

        self.captured_frame_no += 1