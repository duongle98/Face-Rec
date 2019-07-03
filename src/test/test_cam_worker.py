
import unittest
import numpy as np
import pickle

from prod.cam_worker import main
from config import Config
from rabbitmq import RabbitMQ


class CamWorkerTest(unittest.TestCase):

    def test_main(self):
        rb = RabbitMQ(
                        (Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                        (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

        rb.channel.queue_purge(queue=rb.SEND_QUEUE_WORKER)

        main("../data/video/test_sample.mp4", "LOCAL")

        with open("test/unittest_cam_worker.pkl", "rb") as out:
            pickle_messages_dict = pickle.load(out)

        rb_messages_dict = {
                                "display_images": [],
                                "embedding_vectors": [],
                                "recording_area": [],
                                "display_image_bounding_boxes": [],
                            }
        for _ in range(10):

            display_images, embedding_vectors, recording_areas, timestamp, \
                display_image_bounding_boxes = rb.receive_multi_embedding_message(
                                                rb.SEND_QUEUE_WORKER)

            for display_image in display_images:
                rb_messages_dict["display_images"] \
                                .append(np.asarray(display_image))

            for embedding_vector in embedding_vectors:
                rb_messages_dict["embedding_vectors"] \
                                .append(np.asarray(embedding_vector))

            for recording_area in recording_areas:
                rb_messages_dict["recording_area"] \
                                .append(np.asarray(recording_area))

            for display_image_bounding_box in display_image_bounding_boxes:
                rb_messages_dict["display_image_bounding_boxes"] \
                                .append(np.asarray(display_image_bounding_box))

        self.assertEqual(
                            len(pickle_messages_dict["display_images"]),
                            len(rb_messages_dict["display_images"])
                        )

        self.assertEqual(
                            len(pickle_messages_dict["embedding_vectors"]),
                            len(rb_messages_dict["embedding_vectors"])
                        )

        self.assertEqual(
                            len(pickle_messages_dict["recording_area"]),
                            len(rb_messages_dict["recording_area"])
                        )

        self.assertEqual(
                            len(pickle_messages_dict["display_image_bounding_boxes"]),
                            len(rb_messages_dict["display_image_bounding_boxes"])
                        )

        for index in range(len(pickle_messages_dict["display_images"])):
            self.assertEqual(
                                (
                                    rb_messages_dict["display_images"][index] ==
                                    pickle_messages_dict["display_images"][index]
                                ).all(), True
                            )

        for index in range(len(pickle_messages_dict["embedding_vectors"])):
            self.assertEqual(
                                (
                                    rb_messages_dict["embedding_vectors"][index] ==
                                    pickle_messages_dict["embedding_vectors"][index]
                                ).all(), True
                            )

        for index in range(len(pickle_messages_dict["recording_area"])):
            self.assertEqual(
                                (
                                    rb_messages_dict["recording_area"][index] ==
                                    pickle_messages_dict["recording_area"][index]
                                ).all(), True
                            )

        for index in range(len(pickle_messages_dict["display_image_bounding_boxes"])):
            self.assertEqual(
                                (
                                    rb_messages_dict["display_image_bounding_boxes"][index] ==
                                    pickle_messages_dict["display_image_bounding_boxes"][index]
                                ).all(), True
                            )


if __name__ == "__main__":
    unittest.main()
