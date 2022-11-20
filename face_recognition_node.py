import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from deepracer_interfaces_pkg.srv import SetLedCtrlSrv
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

# Nodes in this program
NODE_NAME = 'FaceDetectionNode'

# Topics subscribed/published to in this program
CAMERA_TOPIC_NAME = '/camera_pkg/display_mjpeg'


class FaceDetection(Node):
    def __init__(self):
        # Initializing FaceDetection class
        super().__init__(NODE_NAME)
        self.camera_subscriber = self.create_subscription(Image, CAMERA_TOPIC_NAME, self.custom_controller, 10)
        self.camera_subscriber
        self.bridge = CvBridge()
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_deault.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.detector_params = cv2.SimpleBlobDetector_Params()
        self.detector_params.filterByArea = True
        self.detector_params.maxArea = 1500
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)
        self.attention_queue = deque()

        # Initiate LED ROS service
        self.set_led_ctrl_client = self.create_client(SetLedCtrlSrv, "servo_pkg/set_led_state")

        # Wait for service to become available
        while not self.set_led_ctrl_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"{self.set_led_ctrl_client.srv_name} service not available, waiting again..."
            )

        # Set LED multiplier (provided by AWS)
        self.led_scaling_factor = 39215

        # Set some default colors
        self.red_rgb = [255, 0, 0]
        self.green_rgb = [0, 255, 0]
        self.blue_rgb = [0, 0, 255]
        self.yellow_rgb = [255, 255, 0]

    def cut_eyebrows(self, img):
        """Given an image, cut out the eyebrows from the eyes"""
        height, width = img.shape[:2]
        eyebrow_h = int(height / 4)
        cut_img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
        return cut_img

    def detect_eyes(self, img, cascade):
        """Given an image and a cascade effect, return a nested array of coordinates for respective eyes"""
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = cascade.detectMultiScale(gray_img, 1.3, 5)
        width = np.size(img, 1)
        height = np.size(img, 0)
        left_eye = None
        right_eye = None
        for (x, y, w, h) in eyes:
            if y > height / 2:
                pass
            eyecenter = x + w / 2
            if eyecenter < width / 2:
                left_eye = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            else:
                right_eye = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)

        return left_eye, right_eye

    def detect_faces(self, img, cascade):
        "Given an image and a face cascading effect, return the coordinates of the face"
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cascade.detectMultiScale(gray_img, 1.3, 5)
        if len(coords) > 1:
            biggest = (0, 0, 0, 0)
            for i in coords:
                if i[3] > biggest[3]:
                    biggest = i
            biggest = np.array([i], np.int32)
        elif len(coords) == 1:
            biggest = coords
        else:
            return None
        for (x, y, w, h) in biggest:
            frame = img[y:y + h, x:x + w]

        return frame

    def blob_process(self, img, threshold, detector):
        """Using blur processing, get the pupils of the eyes --threshold is togglable"""
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        keypoints = detector.detect(img)
        return keypoints

    def set_led_color(self, color):
        """Given the attention of the driver, set the color of the tail lights to red or green
            CHANGE TO CONFIGURE MOTOR AS WELL AS LED COLOR LATER"""
        set_led_color_req = SetLedCtrlSrv.Request()
        r = color[0]
        g = color[1]
        b = color[2]
        set_led_color_req.red = r * self.led_scaling_factor
        set_led_color_req.green = g * self.led_scaling_factor
        set_led_color_req.blue = b * self.led_scaling_factor
        self.set_led_ctrl_client.call_async(color)

    def custom_controller(self, data):
        """Main console operations will be run in this module"""

        # Get image feed from camera
        frame = self.bridge.imgmsg_to_cv2(data)
        face_frame = self.detect_faces(frame, self.face_cascade)

        if face_frame is not None:
            eyes = self.detect_eyes(face_frame, self.eye_cascade)

            for eye in eyes:
                if eye is not None:
                    # threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    threshold = r = 38  # timothy threshold value

                    cut_eye = self.cut_eyebrows(eye)
                    keypoints = self.blob_process(cut_eye, threshold, self.detector)

                    blobbed_eye = cv2.drawKeypoints(cut_eye, keypoints, cut_eye, (0, 255, 255),
                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    try:
                        if keypoints[0].pt[0] < (3 / 16 * eye.shape[0]) or keypoints[0].pt[0] > (
                                13 / 16 * eye.shape[0]) or \
                                keypoints[0].pt[1] < (1 / 4 * eye.shape[1]) or keypoints[0].pt[1] > (
                                3 / 4 * eye.shape[1]):
                            attention = False
                        else:
                            attention = True
                    except:
                        attention = False

                if eyes[0] is None or eyes[1] is None:
                    attention = False
                else:
                    attention = True

            if len(eyes) == 0:
                attention = False

        if attention is False:
            self.attention_queue.append(False)
        else:
            self.attention_queue.append(True)
        self.attention_queue.popleft()

        # changes the tail light color based on the attention of the car
        # given that the camera saves frames at ~30 fps, we will check the last 60 frames
        if len(self.attention_queue) > 150 and self.attention_queue[:60].count(False) > 50:
            self.set_led_color(self.red_rgb)
        else:
            self.set_led_color(self.green_rgb)


def main(args=None):
    rclpy.init(args=args)
    centroid_publisher = FaceDetection()
    try:
        rclpy.spin(centroid_publisher)
        centroid_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        centroid_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
        cv2.destroyAllWindows()
        centroid_publisher.destroy_node()
        rclpy.shutdown()
        centroid_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()
