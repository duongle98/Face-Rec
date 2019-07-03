"""
This script contains all handlers for reading image frames from different sources
"""
from config import Config
import os
import glob
from abc import ABCMeta, abstractmethod
import cv2
from scipy import misc


class AbstractFrameReader(metaclass=ABCMeta):
    '''
    Abstract class to provide frame
    '''
    def __init__(self, scale_factor):
        '''
        :param scale_factor: reduce frame factor
        '''
        self._scale_factor = scale_factor

    @abstractmethod
    def next_frame(self):
        '''
        Return next frame, None if empty
        '''
        pass

    def has_next(self):
        '''
        Check if frame is avaiable
        '''
        return True

    def _scale_frame(self, frame):
        '''
        Rescale frame for faster processing
        :param frame: input frame to rescale
        :return frame: resized frame
        '''
        if frame is None:
            return None

        if self._scale_factor > 1:
            frame = cv2.resize(frame, (int(len(frame[0]) / self._scale_factor),
                                       int(len(frame) / self._scale_factor)))
        return frame

    # @abstractmethod
    def release(self):
        '''
        Release this frame reader
        '''
        pass


class RabbitFrameReader(AbstractFrameReader):
    '''
    Read frame from rabbit queue, for register for live image
    '''

    def __init__(self, rb, scale_factor=Config.Frame.SCALE_FACCTOR):
        '''
        :param rb: rabbitmq instance to read new frame from
        :param scale_factor: reduce frame factor
        '''
        self.__rb = rb
        super(RabbitFrameReader, self).__init__(scale_factor)

    def next_frame(self):
        '''
        Read next frame from rabbit, may return None if there is no frame avaiable
        '''
        frame = self.__rb.receive_raw_live_image(self.__rb.RECEIVE_QUEUE_CAPTURE)
        if frame is None:
            frame = self.__rb.receive_raw_live_image(self.__rb.RECEIVE_WEB_CAM)
        return self._scale_frame(frame)

    def release(self):
        pass


class URLFrameReader(AbstractFrameReader):
    """
    Read frame from video stream or from video path or web cam
    """

    # TODO: Should create 3 subclass?
    WEBCAM = 0
    VIDEO_FILE = 1
    IP_STREAM = 2

    def __init__(self, cam_url, scale_factor=Config.Frame.SCALE_FACCTOR):
        '''
        :param cam_url: url for video stream
        :param scale_factor: reduce frame factor
        '''
        if type(cam_url) == int or cam_url.isdigit():
            cam_url = int(cam_url)
            self.__url_type = URLFrameReader.WEBCAM
        elif self.__is_file(cam_url):
            self.__url_type = URLFrameReader.VIDEO_FILE
        else:
            self.__url_type = URLFrameReader.IP_STREAM
        self.__video_capture = cv2.VideoCapture(cam_url)
        super(URLFrameReader, self).__init__(scale_factor)

    def next_frame(self):
        '''
        Return scaled frame from video stream
        '''
        ret, frame = self.__video_capture.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self._scale_frame(frame)
        return frame

    def has_next(self):
        '''
        If the video stream is limited (like video file),
            the frame will return False when it's read all the frame
        else (ip stream or webcam) it will always return True
        '''
        if self.__url_type == URLFrameReader.VIDEO_FILE:
            return self.__video_capture.get(cv2.CAP_PROP_POS_FRAMES) \
                    < self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            return self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) != 0 \
                   and self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) != 0

    def release(self):
        '''
        Release the video source
        '''
        self.__video_capture.release()

    def __is_file(self, cam_url):
        '''
        Check if cam_url is a video in filesystem or an ip stream
        :param cam_url: url to check
        :return True if cam_url exist in file system
        '''
        return os.path.exists(cam_url)

    def get_info(self):
        '''
        Get information
        '''
        return self.__video_capture.get(cv2.CAP_PROP_FPS), \
            self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / self._scale_factor, \
            self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / self._scale_factor, \
            self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)


class DirectoryFrameReader(AbstractFrameReader):
    """
    Read all image from a directory

    >>> frame_reader = DirectoryFrameReader(r'../data/matching/set1', 'jpg'); \
        len(frame_reader._DirectoryFrameReader__image_files)
    9
    >>> for i in range(9):
    ...     _ = frame_reader.next_frame()
    >>> frame_reader.has_next()
    False
    """
    def __init__(self, dir_, ext='jpg', scale_factor=Config.Frame.SCALE_FACCTOR):
        '''
        :param dir_: directory that contains images
        :pram ext: image extension to read
        :param scale_factor: reduce frame factor
        '''
        self.__image_files = glob.glob(os.path.join(dir_, '*.%s' % ext))
        self.__image_files.sort()
        self.__frame_index = 0
        super(DirectoryFrameReader, self).__init__(scale_factor)

    def next_frame(self):
        '''
        Read next image from directory
        '''
        frame = misc.imread(self.__image_files[self.__frame_index])
        self.__frame_index += 1
        return frame

    def has_next(self):
        '''
        Return False when all images have been read
        '''
        return self.__frame_index < len(self.__image_files)

    def release(self):
        '''
        Release all image file
        '''
        self.__frame_index = 0
        self.__image_files = []


if __name__ == '__main__':
    import doctest
    doctest.testmod()
