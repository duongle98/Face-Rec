
import time
import argparse

from config import Config
from rabbitmq import RabbitMQ
from face_detector import MTCNNDetector
from tf_graph import FaceGraph
from face_extractor import FacenetExtractor
from frame_reader import URLFrameReader
from preprocess import Preprocessor, whitening
import utils


def main(cam_url, recording_area):

    rb = RabbitMQ(
                    (Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                    (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))
    detector = MTCNNDetector(FaceGraph())
    frame_reader = URLFrameReader(cam_url)
    edit_image = utils.CropperUtils()
    face_angle = utils.FaceAngleUtils()
    feature_extractor = FacenetExtractor(FaceGraph())
    pre_process = Preprocessor(whitening)

    while frame_reader.has_next():

        embedding_images = []
        embedding_vectors = []
        display_images = []
        display_image_bounding_boxes = []

        frame = frame_reader.next_frame()
        bounding_boxes, points = detector.detect_face(frame)

        for index, bounding_box in enumerate(bounding_boxes):

            if face_angle.is_acceptable_angle(points[:, index]) is True:

                embedding_image = edit_image.crop_face(frame, bounding_box)
                embedding_images.append(embedding_image)

                display_image, display_image_bounding_box = edit_image.crop_display_face(
                                                                        frame, bounding_box)
                display_images.append(display_image)
                display_image_bounding_boxes.append(display_image_bounding_box)

                whitened_image = pre_process.process(embedding_image)
                embedding_vector = feature_extractor.extract_features(whitened_image)

                embedding_vectors.append(embedding_vector)

        if len(embedding_vectors) > 0:

            rb.send_multi_embedding_message(display_images,
                                            embedding_vectors,
                                            recording_area,
                                            time.time(),
                                            display_image_bounding_boxes,
                                            rb.SEND_QUEUE_WORKER)
        else:
            print("No Face Detected")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--cam_url', help='your camera ip address', default=None)
    parser.add_argument(
        '-a', '--area', help='The area that your ip camera is recording at', default=None)
    args = parser.parse_args()
    main(args.cam_url, args.area)
