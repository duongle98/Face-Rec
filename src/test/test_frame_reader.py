import unittest
from frame_reader import URLFrameReader, RabbitFrameReader
from config import ROOT, Config
from rabbitmq import RabbitMQ
import time
from utils import encode_image
import cv2
import numpy as np


class RabbitMQTest(unittest.TestCase):

    def setUp(self):
        if not Config.DEBUG:
            raise('To run RabbitMQTest, DEBUG Mode must be on')
        self.rb = RabbitMQ()
        self.frame_reader = RabbitFrameReader(self.rb)

    def test_no_frame_avaiable(self):
        frame = self.frame_reader.next_frame()
        self.assertIsNone(frame)

    def test_read_frame_from_queue_capture(self):
        sample = sample_image()
        self.rb.send(self.rb.RECEIVE_QUEUE_CAPTURE, encode_image(sample))
        time.sleep(2)
        frame = self.frame_reader.next_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, sample.shape)
        self.assertAlmostEqual(np.sum(frame - sample), 0)

    def test_read_frame_from_queue_web_cam(self):
        sample = sample_image()
        self.rb.send(self.rb.RECEIVE_WEB_CAM, encode_image(sample))
        time.sleep(2)
        frame = self.frame_reader.next_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, sample.shape)
        self.assertAlmostEqual(np.sum(frame - sample), 0)


class URLFrameReaderTest(unittest.TestCase):

    def test_webcam_type_1(self):
        frame_reader = URLFrameReader(cam_url=0)
        self.assertEqual(frame_reader._URLFrameReader__url_type, URLFrameReader.WEBCAM)

    def test_webcam_type_2(self):
        frame_reader = URLFrameReader(cam_url='0')
        self.assertEqual(frame_reader._URLFrameReader__url_type, URLFrameReader.WEBCAM)

    def test_video_type(self):
        frame_reader = URLFrameReader(cam_url='%s//data/video/test-vin.mp4' % ROOT)
        self.assertEqual(frame_reader._URLFrameReader__url_type, URLFrameReader.VIDEO_FILE)

    def test_ip_type(self):
        frame_reader = URLFrameReader(
                    cam_url='rtsp://admin:vietnet@123@14.226.238.27:1024/onvif/profile2/media.smp')
        self.assertEqual(frame_reader._URLFrameReader__url_type, URLFrameReader.IP_STREAM)

    def test_scale_factor(self):
        frame_reader = URLFrameReader(cam_url='%s//data/video/test-vin.mp4' % ROOT, scale_factor=2)
        frame = frame_reader.next_frame()
        self.assertEqual(frame.shape, (540, 960, 3))

    def test_end_of_video(self):
        frame_reader = URLFrameReader(cam_url='%s//data/video/test-vin.mp4' % ROOT)
        for i in range(76):
            self.assertTrue(frame_reader.has_next())
            frame_reader.next_frame()
        self.assertFalse(frame_reader.has_next())

    def test_invalid_url_has_no_frame(self):
        frame_reader = URLFrameReader(cam_url='abc')
        self.assertFalse(frame_reader.has_next())


def sample_image():
    return cv2.imread("%s/data/matching/set1/0_0.jpg" % ROOT)


if __name__ == '__main__':
    unittest.main()
