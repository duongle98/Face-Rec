"""
This script contains all handlers for sending and receiving messages from RabbitMq
"""


import pika
import functools
import numpy as np
import time
from config import Config
from utils import encode_image, decode_image, encode_ndarray, decode_ndarray


def catch_connection_closed(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except pika.exceptions.ConnectionClosed as e:
            print(e, 'from', f.__name__)
            print('re-init rabbit')
            rb = args[0]
            rb.__init__((rb.username, rb.password), (rb.ip, rb.port))
            return f(*args, **kwargs)
    return func


class RabbitMQ:
    """
    Data structure for global queue names and common queue operations
    >>> rb = RabbitMQ()
    >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
    <...
    >>> rb.send('loi-rabbitmq-test-queue', 'hello world')
    message sent
    'hello world'
    >>> rb.receive_once(queue_name='loi-rabbitmq-test-queue')
    ['hello world']
    """

    # VINGROUP cases
    # 1. Recognition server - checkin case
    if Config.DEBUG:
        prefix = "test-"
    else:
        prefix = ""
    VINGROUP_RPC_SERVER = prefix + 'rpc.server.vingroup_api_detection'
    API_SERVER_TO_CENTRAL = prefix + 'eyeq_vingroup_api_to_central'
    # 2. Cam worker
    # a. Rarely used anymore (for register with local webcame)
    RECEIVE_QUEUE_CAPTURE = prefix + 'eyeq-vingroup-capture'
    RECEIVE_WEB_CAM = prefix + 'eyeq-vingroup-cam'

    # b. Send from Cam worker to Central worker
    SEND_QUEUE_WORKER = prefix + 'eyeq_cam_to_central'

    # 3. Central worker
    # queue for correction
    RECEIVE_QUEUE_REGISTER = prefix + 'eyeq-vingroup-register'

    # queue for register from operator - defined above
    # API_SERVER_TO_CENTRAL

    # queue from cam worker to central worker
    RECEIVE_CAM_WORKER_QUEUE = prefix + 'eyeq_cam_to_central'
    RECEIVE_CAM_WORKER_TRACKING_QUEUE = prefix + 'eyeq_cam_to_central-tracking'

    # rarely used (for register case on local cam)
    SEND_QUEUE_LIVE_RESULT = prefix + 'eyeq-vingroup-cam-result'

    # from central to web server
    SEND_QUEUE_DETECTION = prefix + 'eyeq-vingroup-detection'

    # 4. Import profiles in batch
    SEND_QUEUE_IMPORT_REGISTER = prefix + 'eyeq-vingroup-cam-register-result'

    """
    """
    def __init__(self, logins=(Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                 connections=(Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT)):
        """
        >>> rb = RabbitMQ()
        """
        self.username = logins[0]
        self.password = logins[1]
        self.ip = connections[0]
        self.port = connections[1]
        credentials = pika.PlainCredentials(self.username, self.password)
        params = pika.ConnectionParameters(self.ip, self.port, '/', credentials,
                                           frame_max=131072,
                                           socket_timeout=1000,
                                           heartbeat_interval=None)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        self.channel = channel

    @catch_connection_closed
    def send(self, queue_name, msg):
        """
        >>> rb = RabbitMQ()
        >>> rb.send('loi-rabbitmq-test-queue', 'test-string')
        message sent
        'test-string'
        """
        self.channel.basic_publish(exchange='', routing_key=queue_name, body=msg)
        print('message sent')
        return msg

    @catch_connection_closed
    def receive_once(self, queue_name):
        """
        >>> rb = RabbitMQ()
        >>> rb.send('loi-rabbitmq-test-queue', 'test-string')
        message sent
        'test-string'
        >>> rb.receive_once('loi-rabbitmq-test-queue')
        ['test-string']
        """
        method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, no_ack=True)
        if method_frame:
            return body.decode('ascii').split(Config.Rabbit.INTER_SEP)
        else:
            return None

    @catch_connection_closed
    def receive(self, queue_name):
        """
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send('loi-rabbitmq-test-queue', 'test-string')
        message sent
        'test-string'
        >>> rb.send('loi-rabbitmq-test-queue', 'test-string')
        message sent
        'test-string'
        >>> print(rb.receive('loi-rabbitmq-test-queue'))
        [['test-string'], ['test-string']]
        """
        msgs = []
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, no_ack=True)
            if method_frame:
                msgs.append(body.decode('ascii').split(Config.Rabbit.INTER_SEP))
            else:
                break
        return msgs

    """

    """
    # for Cam worker
    def send_multi_embedding_message(self, img_list, emb_arrays, area, time_stamp,
                                     after_padded_bbs, queue_name):
        """
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.send_multi_embedding_message([sample_pixel()], [sample_array()], 'AREA',
        ...                                 1111, ['0_0_0_0'], 'loi-rabbitmq-test-queue')
        ...                                 #doctest: +ELLIPSIS
        message sent
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd...==|1|AREA|1111|0_0_0_0'
        """
        bin_img_list = [encode_image(image) for image in img_list]
        bin_img_list_str = Config.Rabbit.INTRA_SEP.join(bin_img_list)
        # split emb by ? since - can be negative sign
        emb_list = [encode_ndarray(embedding) for embedding in emb_arrays]
        emb_list_str = Config.Rabbit.INTRA_SEP.join(emb_list)
        after_padded_bbs_str = Config.Rabbit.INTRA_SEP.join(after_padded_bbs)

        msg = Config.Rabbit.INTER_SEP.join([bin_img_list_str, emb_list_str, str(area),
                                            str(time_stamp), after_padded_bbs_str])
        return self.send(queue_name, msg)

    def send_capture_result(self, images, queue_name):
        """
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.send_capture_result([sample_pixel(), sample_pixel()], 'loi-rabbitmq-test-queue')
        ...                        #doctest: +ELLIPSIS
        message sent
        'iVBORw0KG...AAABJRU5ErkJggg==?iVBORw0KGgoAA...ADgAF847BZQAAAABJRU5ErkJggg=='
        """
        bin_img_list = [encode_image(image) for image in images]
        bin_img_list_str = Config.Rabbit.INTRA_SEP.join(bin_img_list)
        return self.send(queue_name, bin_img_list_str)

    def receive_raw_live_image(self, queue_name):
        """
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send(queue_name='loi-rabbitmq-test-queue', msg=encode_image(sample_pixel()))
        ...                        #doctest: +ELLIPSIS
        message sent
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACklEQVQIHWMEAg...ABJRU5ErkJggg=='
        >>> rb.receive_raw_live_image('loi-rabbitmq-test-queue')
        array([[[1, 1, 1]]], dtype=uint8)
        """
        msg = self.receive_once(queue_name)
        if msg:
            img = decode_image(msg[0])
            return img
        return msg

    # for Central worker
    def receive_register_image(self, queue_name):
        # face_id | image_id
        """
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send(queue_name='loi-rabbitmq-test-queue', msg='Loi|AREA-1253_1233_66_102-13524796')
        ...         #doctest: +ELLIPSIS
        message sent
        'Loi|AREA-1253_1233_66_102-13524796'
        >>> rb.receive_register_image(queue_name='loi-rabbitmq-test-queue')
        ('Loi', 'AREA-1253_1233_66_102-13524796')
        """
        msg = self.receive_once(queue_name)
        if msg:
            face_id = msg[0].lower().title()
            # print("Face id", face_id)
            image_id = msg[1]
            # image_id = msg[2]
            return (face_id, image_id)
        return None, None

    def receive_embedding_message(self, queue_name):
        """
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send(queue_name='loi-rabbitmq-test-queue',
        ...         msg=Config.Rabbit.INTER_SEP.join([encode_image(sample_pixel()),
        ...                                           encode_ndarray(sample_array()),
        ...                                           'AREA', '1111', '1.5'])) #doctest: +ELLIPSIS
        message sent
        'iVBORw...Jggg==|1|AREA|1111|1.5'
        >>> rb.receive_embedding_message(queue_name='loi-rabbitmq-test-queue')
        (array([[[1, 1, 1]]], dtype=uint8), array([ 1.]), 'AREA', 1111.0, '1.5')
        """
        msg = self.receive_once(queue_name)
        if msg:
            image = decode_image(msg[0])
            emb = decode_ndarray(msg[1])
            area_id = msg[2]
            timestamp = float(msg[3])
            crop_ratio = msg[4]
            return image, emb, area_id, timestamp, crop_ratio
        return None, None, None, None, None

    def receive_multi_embedding_message(self, queue_name):
        """
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send_multi_embedding_message([sample_pixel()], [sample_array()], 'AREA',
        ...                                 1111, ['0_0_0_0'], 'loi-rabbitmq-test-queue')
        ...                                 #doctest: +ELLIPSIS
        message sent
        'iVBORw...Jggg==|1|AREA|1111|0_0_0_0'
        >>> rb.receive_multi_embedding_message(queue_name='loi-rabbitmq-test-queue')
        ([array([[[1, 1, 1]]], dtype=uint8)], [array([ 1.])], 'AREA', 1111.0, ['0_0_0_0'])
        """
        msg = self.receive_once(queue_name)
        if msg:
            img_list = msg[0].split(Config.Rabbit.INTRA_SEP)
            imgs = [decode_image(i) for i in img_list]
            emb_list = msg[1].split(Config.Rabbit.INTRA_SEP)
            embs = [decode_ndarray(e) for e in emb_list]
            area_id = msg[2]
            timestamp = float(msg[3])
            after_padded_bbs = msg[4].split(Config.Rabbit.INTRA_SEP)

            return imgs, embs, area_id, timestamp, after_padded_bbs
        return None, None, None, None, None

    def send_recognition_result(self, human_id, image_id, image, queue_name):
        """
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.send_recognition_result('Loi', 'AREA-1253_1233_66_102-13524796', sample_pixel(),
        ...                            queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        'Loi|AREA-1253_1233_66_102-13524796|iVBORw0KGgoAAAANSUh...AgAADgAF847BZQAAAABJRU5ErkJggg=='
        """
        binaryImg = encode_image(image)
        msg = Config.Rabbit.INTER_SEP.join([str(human_id), str(image_id), binaryImg])
        return self.send(queue_name, msg)

    def send_multi_recognition_result(self, human_ids, image_ids, images, queue_name):
        """
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.send_multi_recognition_result(['Loi', 'Man'],
        ...                                  ['IMAGE_ID_1', 'IMAGE_ID_2'],
        ...                                  [sample_pixel(), sample_pixel()],
        ...                                  'loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        'Loi?Man|IMAGE_ID_1?IMAGE_ID_2|iVBORw...kJggg==?iVBORw...Jggg=='
        """
        human_list_str = Config.Rabbit.INTRA_SEP.join(human_ids)
        image_id_list_str = Config.Rabbit.INTRA_SEP.join(image_ids)
        bin_img_list = [encode_image(i) for i in images]
        bin_img_list_str = Config.Rabbit.INTRA_SEP.join(bin_img_list)
        msg = Config.Rabbit.INTER_SEP.join([human_list_str, image_id_list_str, bin_img_list_str])
        return self.send(queue_name, msg)

    def send_response_from_rpc_to_central(self, queue_name, id_dict, area_id):
        """
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.send_response_from_rpc_to_central('loi-rabbitmq-test-queue',
        ...                                      {'Loi':[(sample_array(),
        ...                                               sample_pixel(),
        ...                                               '0_0_0_0',
        ...                                               562)]},
        ...                                      'CHECKIN-AREA') #doctest: +ELLIPSIS
        message sent
        'iVBORw...Jggg==|1|0|Loi|CHECKIN-AREA|...|0_0_0_0'
        """
        display_face_img_list = []
        emb_list = []
        order_list = []
        padded_bboxes_str_list = []
        id_list = []

        for face_id, embs_faces_list in id_dict.items():
            order_id = 0
            for emb, display_face_image, padded_bbox_str, _ in embs_faces_list:
                emb_list.append(encode_ndarray(emb))
                display_face_img_list.append(encode_image(display_face_image))
                padded_bboxes_str_list.append(padded_bbox_str)
                id_list.append(str(face_id))
                order_list.append(str(order_id))
                order_id += 1

        display_face_img_list_str = Config.Rabbit.INTRA_SEP.join(display_face_img_list)
        emb_list_str = Config.Rabbit.INTRA_SEP.join(emb_list)
        order_list_str = Config.Rabbit.INTRA_SEP.join(order_list)
        id_list_str = Config.Rabbit.INTRA_SEP.join(id_list)
        padded_bboxes_str = Config.Rabbit.INTRA_SEP.join(padded_bboxes_str_list)

        msg = Config.Rabbit.INTER_SEP.join([display_face_img_list_str, emb_list_str,
                                            order_list_str, id_list_str, area_id,
                                            str(int(time.time())), padded_bboxes_str])
        return self.send(queue_name, msg)

    def receive_api_worker_to_central(self, queue_name):
        """
        >>> from utils import sample_array, sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send_response_from_rpc_to_central('loi-rabbitmq-test-queue',
        ...                                      {'Loi':[(sample_array(),
        ...                                               sample_pixel(),
        ...                                               '0_0_0_0',
        ...                                               562)]},
        ...                                      'CHECKIN-AREA') #doctest: +ELLIPSIS
        message sent
        'iVBORw...Jggg==|1|0|Loi|CHECKIN-AREA|...|0_0_0_0'
        >>> rb.receive(queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        [['iVBORw...Jggg==', '1', '0', 'Loi', 'CHECKIN-AREA', '...', '0_0_0_0']]
        """
        msg = self.receive_once(queue_name)
        if msg:
            img_list = msg[0].split(Config.Rabbit.INTRA_SEP)
            if img_list:
                imgs = [decode_image(i) for i in img_list]
            else:
                imgs = []

            emb_list = msg[1].split(Config.Rabbit.INTRA_SEP)
            embs = [decode_ndarray(e) for e in emb_list]
            order_list = msg[2].split(Config.Rabbit.INTRA_SEP)
            id_list = msg[3].split(Config.Rabbit.INTRA_SEP)
            area_id = msg[4]
            timestamp = float(msg[5])
            after_padded_bbs = msg[6].split(Config.Rabbit.INTRA_SEP)

            return imgs, embs, order_list, id_list, area_id, timestamp, after_padded_bbs
        return None, None, None, None, None, None, None

    # for Importing register data
    def send_import_register_data(self, image_face_list, queue_name):
        """
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.send_import_register_data([('Loi', sample_pixel()), ('Man', sample_pixel())],
        ...                              queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        'Loi?iVBORw...Jggg==|Man?iVBORw...Jggg=='
        """
        image_face_str_list = []
        for image_face in image_face_list:
            bin_img_str = encode_image(image_face[1])
            image_face_str_list.append(Config.Rabbit.INTRA_SEP.join([image_face[0], bin_img_str]))
        msg = Config.Rabbit.INTER_SEP.join(image_face_str_list)
        return self.send(queue_name, msg)

    def send_multi_live_reg(self, recog_list, queue_name):
        # recog_list = [(image_id, image, track_id, top_match_ids)...]
        """
        sending top match to web dashboard
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.send_multi_live_reg([('Image_ID_1', sample_pixel(), 'Loi', ['Man', 'Dat', 'Phuc'])],
        ...                        queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        'Image_ID_1?iVBORw...Jggg==?Loi?Man?Dat?Phuc'
        """
        face_msg_list = []
        for recog in recog_list:
            bin_img_str = encode_image(recog[1])
            face_msg = Config.Rabbit.INTRA_SEP.join([recog[0], bin_img_str, recog[2],
                                                    Config.Rabbit.INTRA_SEP.join(recog[3])])
            face_msg_list.append(face_msg)
        msg = Config.Rabbit.INTER_SEP.join(face_msg_list)
        return self.send(queue_name, msg)

    # for Simulating dashboard result
    def receive_recog_result(self, queue_name):
        """
        >>> from utils import sample_pixel
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send_multi_live_reg([('Image_ID_1', sample_pixel(), 'Loi', ['Man', 'Dat', 'Phuc'])],
        ...                        queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        'Image_ID_1?iVBORw...Jggg==?Loi?Man?Dat?Phuc'
        >>> rb.receive_recog_result(queue_name='loi-rabbitmq-test-queue')
        [('Image_ID_1', array([[[1, 1, 1]]], dtype=uint8), 'Loi', ['Man', 'Dat', 'Phuc'])]
        """
        msg = self.receive_once(queue_name)
        recog_list = []
        if msg:
            for recog_str in msg:
                recog_eles = recog_str.split(Config.Rabbit.INTRA_SEP)
                # image_id, image, track_id, top_match_ids
                image_id = recog_eles[0]
                image_str = recog_eles[1]
                image = decode_image(image_str)
                track_id = recog_eles[2]
                top_match_ids = recog_eles[3:]
                recog_list.append((image_id, image, track_id, top_match_ids))

            return recog_list
        return None

    def send_tracking(self, track_tuple, queue_name):
        """
        "send tracking from cam to central" - by Man
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.send_tracking((1, sample_pixel(), sample_array(), 'AREA', 152364, '0_0_0_0'),
        ...                  queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        '1?iVBORw...Jggg==?1?AREA?152364?0_0_0_0'
        """
        # TODO:@loi modify doctest following new format
        fid = str(track_tuple[0])
        image = track_tuple[1]
        emb = track_tuple[2]
        area_id = track_tuple[3]
        timestamp = str(track_tuple[4])
        origin_bb = track_tuple[5]
        angle = str(track_tuple[6])

        bin_img_str = encode_image(image)
        emb_str = encode_ndarray(emb)
        msg_list = [fid, bin_img_str, emb_str, area_id, timestamp, origin_bb, angle]
        msg = Config.Rabbit.INTRA_SEP.join(msg_list)
        return self.send(queue_name, msg)

    def receive_tracking(self, queue_name):
        """
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send_tracking((1, sample_pixel(), sample_array(), 'AREA', 152364, '0_0_0_0'),
        ...                  queue_name='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        message sent
        '1?iVBORw...Jggg==?1?AREA?152364?0_0_0_0'
        >>> rb.receive_tracking(queue_name='loi-rabbitmq-test-queue')
        (1, array([[[1, 1, 1]]], dtype=uint8), array([ 1.]), 'AREA', 152364.0, '0_0_0_0')
        """
        # TODO:@loi modify doctest following new format
        msg = self.receive_once(queue_name)
        if msg:

            track_list = msg[0].split(Config.Rabbit.INTRA_SEP)

            fid = int(track_list[0])
            image_str = track_list[1]
            emb_str = track_list[2]
            area_id = track_list[3]
            timestamp = float(track_list[4])
            origin_bb = track_list[5]
            angle = float(track_list[6])

            image = decode_image(image_str)
            emb = decode_ndarray(emb_str)

            return fid, image, emb, area_id, timestamp, origin_bb, angle
        return None, None, None, None, None, None, None

    def send_multi_embedding_message_with_frame(self, img_list, emb_arrays, area, time_stamp,
                                                after_padded_bbs, frame, origin_bbs, queue_name):
        """
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.send_multi_embedding_message_with_frame([sample_pixel()], [sample_array()], 'AREA',
        ...                                          1111, ['0_0_0_0'], sample_pixel(),
        ...                                          [sample_array()], 'loi-rabbitmq-test-queue')
        ...                                          #doctest: +ELLIPSIS
        message sent
        'iVBORw0KGgoAAAANSUhEUgA...ggg==|1|AREA|1111|0_0_0_0|iVBOR...ggg==|1'
        """
        bin_frame = encode_image(frame)
        bin_img_list = [encode_image(i) for i in img_list]
        bin_img_list_str = Config.Rabbit.INTRA_SEP.join(bin_img_list)

        emb_list = [encode_ndarray(e) for e in emb_arrays]

        after_padded_bbs_str = Config.Rabbit.INTRA_SEP.join(after_padded_bbs)

        origin_bbs_str_list = []
        for origin_bb in origin_bbs:
            origin_bb_str_array = np.array(origin_bb, dtype='U')
            origin_bb_str = '_'.join(origin_bb_str_array)
            origin_bbs_str_list.append(origin_bb_str)
        origin_bbs_str = Config.Rabbit.INTRA_SEP.join(origin_bbs_str_list)
        emb_list_str = Config.Rabbit.INTRA_SEP.join(emb_list)
        msg = Config.Rabbit.INTER_SEP.join([bin_img_list_str, emb_list_str, str(area),
                                            str(time_stamp), after_padded_bbs_str, bin_frame,
                                            origin_bbs_str])
        return self.send(queue_name, msg)

    def receive_multi_embedding_message_with_frame(self, queue_name):
        """
        For getting QA infomation from operator correction in web dashboard
        >>> from utils import sample_pixel, sample_array
        >>> rb = RabbitMQ()
        >>> rb.channel.queue_purge(queue='loi-rabbitmq-test-queue') #doctest: +ELLIPSIS
        <...
        >>> rb.send_multi_embedding_message_with_frame([sample_pixel()], [sample_array()], 'AREA',
        ...                                          1111, ['0_0_0_0'], sample_pixel(),
        ...                                          [sample_array()], 'loi-rabbitmq-test-queue')
        ...                                          #doctest: +ELLIPSIS
        message sent
        'iVBORw0KGgoAAAANSUhEUgA...ggg==|1|AREA|1111|0_0_0_0|iVBOR...ggg==|1'
        >>> rb.receive_multi_embedding_message_with_frame(queue_name='loi-rabbitmq-test-queue')
        ... #doctest: +ELLIPSIS
        ([array([[[1, 1, 1]]]...], [array([ 1.])], ..., 1111.0, ['0_0_0_0'],..., array([[1]]...))
        """
        msg = self.receive_once(queue_name)
        if msg:
            img_list = msg[0].split(Config.Rabbit.INTRA_SEP)
            imgs = [decode_image(i) for i in img_list]
            emb_list = msg[1].split(Config.Rabbit.INTRA_SEP)
            embs = [decode_ndarray(e) for e in emb_list]

            area_id = msg[2]
            timestamp = float(msg[3])
            after_padded_bbs = msg[4].split(Config.Rabbit.INTRA_SEP)

            bin_frame = msg[5]
            frame = decode_image(bin_frame)

            origin_bbs_str = msg[6].split(Config.Rabbit.INTRA_SEP)
            origin_bbs_list = [np.array(bb.split('_'), dtype=np.int32) for bb in origin_bbs_str]
            origin_bbs = np.vstack(origin_bbs_list)

            return imgs, embs, area_id, timestamp, after_padded_bbs, frame, origin_bbs
        return None, None, None, None, None, None, None
