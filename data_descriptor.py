from __init__ import *
"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type                 Describes the type of object: 'Car', 'Pedestrian', ‘Vehicles’
                            ‘Vegetation’, 'TrafficSigns', etc.
   3    velocity             velocity of the object, returns three absolute values of the components x, y and z.
   3    acceleration         acceleration of the objects, returns three absolute values of the components x, y and z.
   3    angular_velocity     angular_velocity of the objects, returns three absolute values of the components x, y and z.
"""

class CarlaDescriptor:
    def __init__(self):
        self.type = None
        self.velocity = None
        self.acceleration = None
        self.angular_velocity = None


    def set_type(self, obj_type: str):
        self.type = obj_type

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_acceleration(self, acceleration):
        self.acceleration = acceleration

    def set_angular_velocity(self, angular_velocity):
        self.angular_velocity = angular_velocity

    def __str__(self):
        return "{} {} {} {}".format(self.type, self.velocity, self.angular_velocity, self.acceleration)

"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Pedestrian', ‘Vehicles’
                     ‘Vegetation’, 'TrafficSigns', etc.
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""

class KittiDescriptor(CarlaDescriptor):
    """
    Kitti格式的label类
    """
    def __init__(self, type=None, bbox=None, dimensions=None, location=None, rotation_y=None, extent=None):
        super().__init__()

        self.type = type
        self.truncated = 0
        self.occluded = 0
        self.alpha = -10
        self.bbox = bbox
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.extent = extent

    def set_truncated(self, truncated: float):
        assert 0 <= truncated <= 1, """Truncated must be Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries """
        self.truncated = truncated

    def set_occlusion(self, occlusion: int):
        assert occlusion in range(0, 4), """Occlusion must be Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown"""
        self.occluded = occlusion

    def set_alpha(self, alpha: float):
        assert -pi <= alpha <= pi, "Alpha must be in range [-pi..pi]"
        self.alpha = alpha

    def set_bbox(self, bbox: List[int]):
        assert len(bbox) == 4, """ Bbox must be 2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates (two points)"""
        self.bbox = bbox

    def set_3d_object_dimensions(self, bbox_extent):
        # Bbox extent consists of x,y and z.
        # The bbox extent is by Carla set as
        # x: length of vehicle (driving direction)
        # y: to the right of the vehicle
        # z: up (direction of car roof)
        # However, Kitti expects height, width and length (z, y, x):
        height, width, length = bbox_extent.z, bbox_extent.x, bbox_extent.y
        # Since Carla gives us bbox extent, which is a half-box, multiply all by two
        self.extent = (height, width, length)
        self.dimensions = "{} {} {}".format(2*height, 2*width, 2*length)

    def set_3d_object_location(self, obj_location):
        """
            carla camera coordinate to kitti caemra coordinate
            carla x y z
            kitti z x -y
            z
            ▲   ▲ x
            |  /
            | /
            |/____> y
            However, the camera coordinate system for KITTI is defined as
                ▲ z
               /
              /
             /____> x
            |
            |
            |
            ▼
            y
            Carla: X   Y   Z
            KITTI:-X  -Y   Z
        """
        # Object location is four values (x, y, z, w). We only care about three of them (xyz)
        x, y, z = [float(x) for x in obj_location][0:3]
        assert None not in [
            self.extent, self.type], "Extent and type must be set before location!"

        if self.type == "Pedestrian":
            # Since the midpoint/location of the pedestrian is in the middle of the agent, while for car it is at the bottom
            # we need to subtract the bbox extent in the height direction when adding location of pedestrian.
            z -= self.extent[0]

        self.location = " ".join(map(str, [y, -z, x]))

    def set_rotation_y(self, rotation_y: float):
        assert - \
            pi <= rotation_y <= pi, "Rotation y must be in range [-pi..pi] - found {}".format(
                rotation_y)
        self.rotation_y = rotation_y

    def full_info(self):
        if self.bbox is None:
            bbox_format = " "
        else:
            bbox_format = " ".join([str(x) for x in self.bbox])
        return "{} {} {} {} {} {} {} {} {} {} {}".format(self.type, self.truncated, self.occluded,
                                                         self.alpha, bbox_format, self.dimensions, self.location,
                                                         self.rotation_y, self.velocity, self.angular_velocity, self.acceleration)
    def carla_info(self):
        return "{} {} {} {}".format(self.type, self.velocity, self.acceleration, self.angular_velocity)
    
    def load_kitti_str(self, info_list):
        num_info = len(info_list)
        assert num_info == 15, "FORMAT ERROR: String should be in kitti format"
        self.set_type(info_list[0])
        self.set_truncated(float(info_list[1]))
        self.set_occlusion(int(info_list[2]))
        self.alpha = float(info_list[3])
        self.set_bbox(list(map(int, info_list[4:8])))
        self.dimensions = " ".join(info_list[8:11])
        self.location = " ".join(info_list[11:14])
        self.set_rotation_y(float(info_list[14]))
    
    def load_full_str(self, info_list):
        num_info = len(info_list)
        assert num_info == 24, "FORMAT ERROR: String should be in full format"
        self.load_kitti_str(info_list[:15])
        self.velocity = " ".join(info_list[15:18])
        self.angular_velocity = " ".join(info_list[18:21])
        self.acceleration = " ".join(info_list[21:24])

    def load_from_str(self, info):
        info_list = info.split(" ")
        num_info = len(info_list)
        if num_info == 15:
            self.load_kitti_str(info_list)
        elif num_info == 24:
            self.load_full_str(info_list)
        else:
            print("ONLY FOR KITTI")



    def __str__(self):
        """ Returns the kitti formatted string of the datapoint if it is valid (all critical variables filled out), else it returns an error."""
        if self.bbox is None:
            bbox_format = " "
        else:
            bbox_format = " ".join([str(x) for x in self.bbox])

        # format of kitti track dataset
        return "{} {} {} {} {} {} {} {}".format(self.type, self.truncated, self.occluded,
                                                         self.alpha, bbox_format, self.dimensions, self.location,
                                                         self.rotation_y)


class LidarDescriptor(CarlaDescriptor):
    """
    Kitti-like label class
    """
    def __init__(self, type=None, bbox=None, dimensions=None, location=None, rotation_y=None, extent=None):
        super().__init__()

        self.type = type
        # 3D bounding box in lidar coord: A-B-C-D-A'-B'-C'-D'
        # xA, yA, xB, yB, xC, yC, xD, yD, z1, z2
        self.bbox = bbox 
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.extent = extent
    
    def __str__(self):
        bbox_format = " ".join([str(x) for x in self.bbox])

        return "{} {} {} {} {}".format(self.type, bbox_format, self.velocity, self.angular_velocity,
                                                         self.acceleration)

    def load_from_str(self, info):
        info_list = info.split(" ")
        num_info = len(info_list)
        assert num_info == 20

        self.type = info_list[0]
        self.bbox = list(map(float, info_list[1:11]))
        
        self.velocity = " ".join(info_list[11:14])
        self.angular_velocity = " ".join(info_list[14:17])
        self.acceleration = " ".join(info_list[17:20])