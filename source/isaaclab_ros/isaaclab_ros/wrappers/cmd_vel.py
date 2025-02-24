import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('Cmd_Vel_Subscriber')

        # Initialize base_command variable
        self.cmd_vel = [0.0, 0.0, 0.0]

        # Create subscriber to cmd_vel
        self.create_subscription(Twist, 'go2/cmd_vel', self.cmd_vel_cb, 10)

    def cmd_vel_cb(self, msg):
        """Callback function that updates base_command when a new Twist message is received."""
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]