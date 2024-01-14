"""
This file contains all the methods responsible for saving the generated data in the correct output format.

"""

from __init__ import *

from data_utils import *
from data_descriptor import KittiDescriptor, LidarDescriptor

with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)

MAX_RENDER_DEPTH_IN_METERS = cfg["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MIN_VISIBLE_VERTICES_FOR_RENDER"]
MAX_OUT_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
WINDOW_WIDTH = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]

def depth2array(depth_data):
    array = np.frombuffer(depth_data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (depth_data.height, depth_data.width, 4))  # RGBA format
    array = array[:, :, :3]  # Take only RGB
    array = array[:, :, ::-1]  # BGR
    array = array.astype(np.float32)  # 2ms
    gray_depth = ((array[:, :, 0] + array[:, :, 1] * 256.0 + array[:, :, 2] * 256.0 * 256.0) / (
            (256.0 * 256.0 * 256.0) - 1))  # 2.5ms
    gray_depth = 1000 * gray_depth
    return gray_depth

def img2array(image_data):
    img_array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
    img_array = np.reshape(img_array, (image_data.height, image_data.width, 4))
    img_array = img_array[:, :, :3][:, :, ::-1]
    return img_array

def save_ref_files(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        logging.info("Wrote reference files to %s", path)


def save_image_data(filename, image):
    logging.info("Wrote image data to %s", filename)
    image.save_to_disk(filename)

def save_depth_data(filename, depth_data):
    depth_array = depth2array(depth_data)
    logging.info("Wrote depth data to %s", filename)
    np.save(filename, depth_array)

def save_bbox_image_data(filename, image):
    im = Image.fromarray(image)
    im.save(filename)

def save_lidar_data(filename, point_cloud, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.
        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        point_cloud = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
        point_cloud = point_cloud[:, :-1]

        lidar_array = [[point[0], -point[1], point[2], 1.0]
                       for point in point_cloud]
        lidar_array = np.array(lidar_array).astype(np.float32)
        logging.debug("Lidar min/max of x: {} {}".format(
                      lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of y: {} {}".format(
                      lidar_array[:, 1].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of z: {} {}".format(
                      lidar_array[:, 2].min(), lidar_array[:, 0].max()))
        lidar_array.tofile(filename)


def save_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)

def save_full_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([point.full_info() for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)

def save_carla_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([point.carla_info() for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)

def save_calibration_matrices(transform, filename, intrinsic_mat):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)

    camera_transform = transform[0]
    lidar_transform = transform[1]
    # pitch yaw rool
    b = math.radians(lidar_transform.rotation.pitch-camera_transform.rotation.pitch)
    x = math.radians(lidar_transform.rotation.yaw-camera_transform.rotation.yaw)
    a = math.radians(lidar_transform.rotation.roll-lidar_transform.rotation.roll)
    R0 = np.identity(3)

    TR = np.array([[math.cos(b) * math.cos(x), math.cos(b) * math.sin(x), -math.sin(b)],
                    [-math.cos(a) * math.sin(x) + math.sin(a) * math.sin(b) * math.cos(x),
                     math.cos(a) * math.cos(x) + math.sin(a) * math.sin(b) * math.sin(x), math.sin(a) * math.cos(b)],
                    [math.sin(a) * math.sin(x) + math.cos(a) * math.sin(b) * math.cos(x),
                     -math.sin(a) * math.cos(x) + math.cos(a) * math.sin(b) * math.sin(x), math.cos(a) * math.cos(b)]])
    TR_velodyne = np.dot(TR, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

    TR_velodyne = np.dot(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]), TR_velodyne)

    '''
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    '''
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)

def save_rgb_image(filename, image):
    im = Image.fromarray(image)
    im.save(filename)

def save_objects(kitti_path, carla_path, all_path, lidar_path, depth_data, agent, object_data, intrinsic, extrinsic, world_2_lidar):
    # TODO: add environment objects
    # environment_actors = world.world.get_environment_objects(carla.CityObjectLabel.Any)
    depth_image = depth2array(depth_data)
    all_objects = []
    visible_objects = []
    lidar_objects = []
    for obj in object_data:
        kitti_point, lidar_bbox, flag = is_visible_by_bbox(agent, obj, depth_image, intrinsic, extrinsic, world_2_lidar)
        if flag:
            visible_objects.append(kitti_point)
        all_objects.append(kitti_point)
        lidar_objects.append(lidar_bbox)
    save_carla_label_data(carla_path, visible_objects) # save object in kitti format in Camera data
    save_label_data(kitti_path, visible_objects) # save object in kitti-like format in Camera data
    save_full_label_data(all_path, all_objects) # save object in kitti-like format in Lidar data
    save_label_data(lidar_path, lidar_objects) # save object with bounding box's corners coord in Lidar data

# agent: dict{get_transform}
# obj: dict{transform, bounding_box, type}
def is_visible_by_bbox(agent, obj, depth_image, intrinsic, extrinsic, world_2_lidar):
    obj_transform = obj["transform"]
    obj_bbox = obj["bounding_box"]
    vertices_pos2d = bbox_2d_from_agent(intrinsic, extrinsic, obj_bbox, obj_transform, 1)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(vertices_pos2d, depth_image)
    
    obj_tp = obj["type"]
    midpoint = midpoint_from_agent_location(obj_transform.location, extrinsic)
    bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
    rotation_y = get_relative_rotation_y(agent["transform"].rotation, obj_transform.rotation) % math.pi
    ext = obj_bbox.extent
    truncated = num_vertices_outside_camera / 8
    if num_visible_vertices >= 6:
        occluded = 0
    elif num_visible_vertices >= 4:
        occluded = 1
    else:
        occluded = 2

    g = 9.800002098083496
    velocity = obj["velocity"]
    acceleration = obj["acceleration"]
    angular_velocity = obj["angular_velocity"]

    velocity = "{} {} {}".format(velocity.x, velocity.y, velocity.z)
    acceleration = "{} {} {}".format(acceleration.x, acceleration.y, acceleration.z)
    angular_velocity = "{} {} {}".format(angular_velocity.x, angular_velocity.y, angular_velocity.z + g)
    # draw_3d_bounding_box(rgb_image, vertices_pos2d)

    kitti_data = KittiDescriptor()
    kitti_data.set_truncated(truncated)
    kitti_data.set_occlusion(occluded)
    kitti_data.set_bbox(bbox_2d)
    kitti_data.set_3d_object_dimensions(ext)
    kitti_data.set_type(obj_tp)
    kitti_data.set_3d_object_location(midpoint)
    kitti_data.set_rotation_y(rotation_y)
    kitti_data.set_velocity(velocity)
    kitti_data.set_acceleration(acceleration)
    kitti_data.set_angular_velocity(angular_velocity)

    lidar_data = LidarDescriptor()
    lidar_data.set_type(obj_tp)
    lidar_data.set_velocity(velocity)
    lidar_data.set_acceleration(acceleration)
    lidar_data.set_angular_velocity(angular_velocity)
    coords = create_bb_points(obj_bbox)
    world_coords = vehicle_to_world(coords, obj_bbox, obj_transform)
    lidar_coords = np.dot(world_2_lidar, world_coords)
    lidar_coords[1, :] = -lidar_coords[1, :]
    # xA, yA, xB, yB, xC, yC, xD, yD, z1, z2
    lidar_data.bbox = [lidar_coords[0, 0], lidar_coords[1, 0],
                        lidar_coords[0, 1], lidar_coords[1, 1],
                        lidar_coords[0, 2], lidar_coords[1, 2],
                        lidar_coords[0, 3], lidar_coords[1, 3],
                        lidar_coords[2, 0], lidar_coords[2, 4]]
    

    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER:
        return kitti_data, lidar_data, True
    else:
        return kitti_data, lidar_data, False