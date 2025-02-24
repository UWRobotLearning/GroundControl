import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomPublisher(Node):
    def __init__(self, ns: str = ""):
        super().__init__('odom_publisher')

        # Create publisher to odom
        self.odom_pub = self.create_publisher(Odometry, f'{ns}/platform/odom', 10)
    
    def publish_odom(self, pos, quat, lin_vel, ang_vel):
        msg = Odometry()
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])
        msg.pose.pose.orientation.w = float(quat[0])
        msg.twist.twist.linear.x = float(lin_vel[0])
        msg.twist.twist.linear.y = float(lin_vel[1])
        msg.twist.twist.linear.z = float(lin_vel[2])
        msg.twist.twist.angular.x = float(ang_vel[0])
        msg.twist.twist.angular.y = float(ang_vel[1])
        msg.twist.twist.angular.z = float(ang_vel[2])
        self.odom_pub.publish(msg)
