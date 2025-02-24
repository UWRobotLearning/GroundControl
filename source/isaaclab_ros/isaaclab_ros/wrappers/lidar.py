import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
import tf2_ros as tf2

# isaaclab imports
from isaaclab_ros.config import LidarROSCfg
from isaaclab.managers import SceneEntityCfg

class LidarPublisher(Node):
    '''
    Class that takes Lidar data directly from the IsaacLab scene and republishes them as ROS2 type messages.
    '''
    def __init__(self,
                 lidar_cfg: LidarROSCfg,
                 ns: str = "",
                 **kwargs):
        
        super().__init__('point_cloud_publisher')
        self.lidar_cfg = lidar_cfg

        if self.lidar_cfg.message_type == "PointCloud2":
            self.publisher_ = self.create_publisher(PointCloud2, f'{ns}/{self.lidar_cfg.topic_name}', 10)
        elif self.lidar_cfg.message_type == "LaserScan":
            self.publisher_ = self.create_publisher(LaserScan, f'{ns}/{self.lidar_cfg.topic_name}', 10)
        else:
            raise ValueError("Invalid message type for lidar sensor")

        # Create tf2 objects for tranforms
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self)

    def publish_pointcloud(self, lidar_points):
        # points = self._transform_to_base(lidar_points)
        msg = PointCloud2()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = 1
        msg.width = len(lidar_points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(lidar_points)
        msg.is_dense = True
        msg.data = np.array(lidar_points).tobytes()
        self.publisher_.publish(msg)

    ## TODO: Implement this function
    def publish_laserscan(self, lidar_cfg, scene):
        return

    # def _transform_to_base(self, points, source_frame: str = "go2/map", traget_frame: str = "base_link"):
    #     transformed_points = []
    #     try:
    #         print(rclpy.time.Time())
    #         transform = self.tf_buffer.lookup_transform(traget_frame, source_frame, rclpy.time.Time())
    #     except tf2.TransformException as ex:
    #         self.get_logger().info(f'Could not transform {source_frame} to {traget_frame}: {ex}')
    #         return
    #     for point in points:
    #         point_msg = PointStamped()
    #         point_msg.header.frame_id = source_frame
    #         point_msg.point.x = point[0]
    #         point_msg.point.y = point[1]
    #         point_msg.point.z = point[2]
    #         transformed_point = self.tf_buffer.transform(point_msg, traget_frame)
    #         transformed_points.append([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])
    #     return transformed_points