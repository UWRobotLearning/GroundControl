import rclpy
from rclpy.node import Node
import tf2_ros as tf2
from geometry_msgs.msg import Twist, PointStamped

from ..config import IsaacLabRosCfg
from ..wrappers import LidarPublisher, OdomPublisher, ImuPublisher

class IsaacLabRos(Node):
    def __init__(self,
                 scene,
                 isaac_lab_ros_cfg: IsaacLabRosCfg,
                 **kwargs):
        super().__init__('isaaclab_ros')

        self.isaaclab_ros_cfg = isaac_lab_ros_cfg
        ns = self.isaaclab_ros_cfg.name
        self.get_robot_data(scene)

        # Initialize base_command variable
        self.cmd_vel = [0.0, 0.0, 0.0]

        # Create subscriber to cmd_vel
        self.create_subscription(Twist, 'go2/cmd_vel', self.cmd_vel_cb, 10)

        self.imu_pub = ImuPublisher(ns=ns)
        self.odom_pub = OdomPublisher(ns=ns)
    
    def cmd_vel_cb(self, msg):
        """Callback function that updates base_command when a new Twist message is received."""
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]

    def get_robot_data(self, scene):
        self.robot_position = scene._articulations["robot"]._data.root_pos_w[0]
        self.robot_orientation = scene._articulations["robot"]._data.root_quat_w[0]
        self.robot_lin_vel = scene._articulations["robot"]._data.root_lin_vel_b[0]
        self.robot_ang_vel = scene._articulations["robot"]._data.root_ang_vel_b[0]
        
    def publish(self):
        # pass
        self.odom_pub.publish_odom(self.robot_position, self.robot_orientation, self.robot_lin_vel, self.robot_ang_vel)

