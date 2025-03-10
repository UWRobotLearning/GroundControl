import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImagePublisher(Node):
    def __init__(self, ns: str = ""):
        super().__init__('camera_publisher')

        self.bridge = CvBridge()

        # Create publisher to odom
        self.rgb_pub = self.create_publisher(Image, f'{ns}/image/color', 10)
        self.depth_pub = self.create_publisher(Image, f'{ns}/image/depth', 10)

    def publish_depth(self, depth_image):
        print(depth_image)
        msg = Image()
        msg.header.frame_id = "camera"
        msg.height = depth_image.shape[0]
        msg.width = depth_image.shape[1]
        msg.data = depth_image
        self.depth_pub.publish(msg)
    
    def publish_rgb(self, rgb_image):
        image_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
        msg = Image()
        msg.header.frame_id = "camera"
        msg.height = rgb_image.shape[0]
        msg.width = rgb_image.shape[1]
        msg.data = rgb_image
        self.rgb_pub.publish(image_msg)

