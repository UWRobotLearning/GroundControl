import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuPublisher(Node):
    def __init__(self, ns: str = ""):
        super().__init__('odom_publisher')

        # Create publisher to odom
        self.odom_pub = self.create_publisher(Imu, f'{ns}/imu/data', 10)
    
    def publish_imu(self, imu_obs):
        msg = Imu()
        msg.orientation.x = float(imu_obs[1])
        msg.orientation.y = float(imu_obs[2])
        msg.orientation.z = float(imu_obs[3])
        msg.orientation.w = float(imu_obs[0])
        msg.angular_velocity.x = float(imu_obs[4]) 
        msg.angular_velocity.y = float(imu_obs[5])
        msg.angular_velocity.z = float(imu_obs[6])
        msg.linear_acceleration.x = float(imu_obs[7])
        msg.linear_acceleration.y = float(imu_obs[8])
        msg.linear_acceleration.z = float(imu_obs[9])
        self.odom_pub.publish(msg)
