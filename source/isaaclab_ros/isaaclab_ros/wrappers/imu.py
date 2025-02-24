import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuPublisher(Node):
    def __init__(self, ns: str = ""):
        super().__init__('odom_publisher')

        # Create publisher to odom
        self.odom_pub = self.create_publisher(Imu, f'{ns}/imu/data', 10)
    
    def publish_imu(self, quat, ang_vel, acc):
        msg = Imu()
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        msg.orientation.w = float(quat[0])
        msg.angular_velocity.x = float(ang_vel[0]) 
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])
        msg.linear_acceleration.x = float(acc[0])
        msg.linear_acceleration.y = float(acc[1])
        msg.linear_acceleration.z = float(acc[2])
        self.odom_pub.publish(msg)
