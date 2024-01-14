from __init__ import *

with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)

MAX_RENDER_DEPTH_IN_METERS = cfg["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MIN_VISIBLE_VERTICES_FOR_RENDER"]
MAX_OUT_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
WINDOW_WIDTH = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]

def obj_type(obj):
    if isinstance(obj, carla.EnvironmentObject):
        return obj.type
    else:
        if obj.type_id.find('walker') != -1:
            return 'Pedestrian'
        if obj.type_id.find('vehicle') != -1:
            return 'Car'
        return None

def get_relative_rotation_y(agent_rotation, obj_rotation):
    """ 返回actor和camera在rotation yaw的相对角度 """

    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return degrees_to_radians(rot_agent - rot_car)


def bbox_2d_from_agent(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    bbox = vertices_from_extension(obj_bbox.extent)
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        bbox = transform_points(bbox_transform, bbox)
    else:
        box_location = carla.Location(obj_bbox.location.x-obj_transform.location.x,
                                      obj_bbox.location.y-obj_transform.location.y,
                                      obj_bbox.location.z-obj_transform.location.z)
        box_rotation = obj_bbox.rotation
        bbox_transform = carla.Transform(box_location, box_rotation)
        bbox = transform_points(bbox_transform, bbox)
    # 获取bbox在世界坐标系下的点的坐标
    bbox = transform_points(obj_transform, bbox)
    # 将世界坐标系下的bbox八个点转换到二维图片中
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def vertices_from_extension(ext):
    """ 以自身为原点的八个点的坐标 """
    return np.array([
        [ext.x, ext.y, ext.z],  # Top left front
        [- ext.x, ext.y, ext.z],  # Top left back
        [ext.x, - ext.y, ext.z],  # Top right front
        [- ext.x, - ext.y, ext.z],  # Top right back
        [ext.x, ext.y, - ext.z],  # Bottom left front
        [- ext.x, ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])


def transform_points(transform, points):
    """ 作用：将三维点坐标转换到指定坐标系下 """
    # 转置
    points = points.transpose()
    # [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]  (4,8)
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # transform.get_matrix() 获取当前坐标系向相对坐标系的旋转矩阵
    points = np.mat(transform.get_matrix()) * points
    # 返回前三行
    return points[0:3].transpose()


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """将bbox在世界坐标系中的点投影到该相机获取二维图片的坐标和点的深度"""
    vertices_pos2d = []
    for vertex in bbox:
        # 获取点在world坐标系中的向量
        pos_vector = vertex_to_world_vector(vertex)
        # 将点的world坐标转换到相机坐标系中
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 将点的相机坐标转换为二维图片的坐标
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # 点实际的深度
        vertex_depth = pos2d[2]
        # 点在图片中的坐标
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """ 以carla世界向量（X，Y，Z，1）返回顶点的坐标 （4,1）"""
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def calculate_occlusion_stats(vertices_pos2d, depth_image):
    """ 作用：筛选bbox八个顶点中实际可见的点 """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # 点在可见范围中，并且没有超出图片范围
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), vertex_depth, depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def point_in_canvas(pos):
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False


def point_is_occluded(point, vertex_depth, depth_image):
    y, x = map(int, point)
    from itertools import product
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy + y, dx + x)):
            # 判断点到图像的距离是否大于深对应深度图像的深度值
            if depth_image[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # 当四个邻居点都大于深度图像值时，点被遮挡。返回true
    return all(is_occluded)


def midpoint_from_agent_location(location, extrinsic_mat):
    """ 将agent在世界坐标系中的中心点转换到相机坐标系下 """
    midpoint_vector = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


def proj_to_camera(pos_vector, extrinsic_mat):
    """ 作用：将点的world坐标转换到相机坐标系中 """
    # inv求逆矩阵
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def proj_to_2d(camera_pos_vector, intrinsic_mat):
    """将相机坐标系下的点的3d坐标投影到图片上"""
    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    # normalize the 2D points
    pos2d = np.array([
        pos2d[0] / pos2d[2],
        pos2d[1] / pos2d[2],
        pos2d[2]
    ])
    return pos2d


def filter_by_distance(data_dict, dis):
    environment_objects = data_dict["environment_objects"]
    actors = data_dict["actors"]
    for agent,_ in data_dict["agents_data"].items():
        data_dict["environment_objects"] = [obj for obj in environment_objects if
                                            distance_between_locations(obj.transform.location, agent.get_location())
                                            <dis]
        data_dict["actors"] = [act for act in actors if
                                            distance_between_locations(act.get_location(), agent.get_location())<dis]


def distance_between_locations(location1, location2):
    return math.sqrt(pow(location1.x-location2.x, 2)+pow(location1.y-location2.y, 2))

def calc_projected_2d_bbox(vertices_pos2d):
    """ 根据八个顶点的图片坐标，计算二维bbox的左上和右下的坐标值 """
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_x, min_y, max_x, max_y]

def degrees_to_radians(degrees):
    return degrees * math.pi / 180

def _create_bb_points(ext_x, ext_y, ext_z):
    """
    Returns 3D bounding box for a vehicle.
    """

    cords = np.zeros((8, 4))
    cords[0, :] = np.array([ext_x, ext_y, -ext_z, 1])
    cords[1, :] = np.array([-ext_x, ext_y, -ext_z, 1])
    cords[2, :] = np.array([-ext_x, -ext_y, -ext_z, 1])
    cords[3, :] = np.array([ext_x, -ext_y, -ext_z, 1])
    cords[4, :] = np.array([ext_x, ext_y, ext_z, 1])
    cords[5, :] = np.array([-ext_x, ext_y, ext_z, 1])
    cords[6, :] = np.array([-ext_x, -ext_y, ext_z, 1])
    cords[7, :] = np.array([ext_x, -ext_y, ext_z, 1])
    return cords

def create_bb_points(vehicle_bb):
    """
    Returns 3D bounding box for a vehicle.
    """
    extent = vehicle_bb.extent
    return _create_bb_points(extent.x, extent.y, extent.z)

def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.identity(4)
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def vehicle_to_world(cords, vehicle_bb, vehicle_trans):
    """
    Transforms coordinates of a vehicle bounding box to world.
    """

    bb_transform = carla.Transform(vehicle_bb.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle_trans)
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

def eulerAngles2rotationMat(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = R_x.dot(R_y.dot(R_z))
    return R

def bbox_cam2velo(obj, cam2velo):
    height, width, length = map(float, obj.dimensions.split())
    ext_h = height / 2
    ext_w = width / 2
    ext_l = length / 2
    ext_h += ext_h / 10
    ext_w += ext_w / 10
    ext_l += ext_l / 10
    location = np.array(list(map(float, obj.location.split())))
    rotation = obj.rotation_y
    coords = _create_bb_points(ext_w, ext_l, ext_h)
    coords[:, 2] += ext_h
    location = np.r_[location, np.array([1])].reshape((4, 1))
    location = cam2velo @ location
    theta = [0., 0., -rotation]
    R = eulerAngles2rotationMat(theta)
    T = np.r_[R, np.zeros((1, 3))]
    T = np.c_[T, location]
    lidar_coords = T @ coords.T

    # xA, yA, xB, yB, xC, yC, xD, yD, z1, z2
    bbox = [lidar_coords[0, 0], lidar_coords[1, 0],
            lidar_coords[0, 1], lidar_coords[1, 1],
            lidar_coords[0, 2], lidar_coords[1, 2],
            lidar_coords[0, 3], lidar_coords[1, 3],
            lidar_coords[2, 0], lidar_coords[2, 4]]
    Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, z1, z2 = bbox
    corners_3d = [[Ax, Ay, z1], [Bx, By, z1], [Cx, Cy, z1], [Dx, Dy, z1],
                  [Ax, Ay, z2], [Bx, By, z2], [Cx, Cy, z2], [Dx, Dy, z2]]
    corners_3d = np.array(corners_3d)
    return corners_3d