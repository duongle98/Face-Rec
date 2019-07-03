"""
This script is to provide a face detection and mapping for vingroup check0in case
"""


import os
import time
import glob
import operator
import numpy as np
from scipy import misc
import preprocess
from config import Config
from tf_graph import FaceGraph
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from rabbitmq import RabbitMQ
from utils import get_area, CropperUtils, show_frame


rb = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
              (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))
face_graph = FaceGraph()
detector = MTCNNDetector(face_graph)
feature_extractor = FacenetExtractor(face_graph)


def main():
    """
    >>> Config.DEBUG = True
    >>> main() #doctest: +ELLIPSIS
    RPC-Server debug mode
    id_list [0, 1, 2, 3, 4, 5, 6]
    found id: 0 ; min dist: 0.103708323539
    id_list [0, 1, 2, 3, 4, 5, 6, 0]
    found id: 1 ; min dist: 0.245194554133
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1]
    found id: 2 ; min dist: 0.171797965695
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    found id: 6 ; min dist: 0.326774027331
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6]
    found id: 5 ; min dist: 0.317452337283
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6, 5]
    found id: 0 ; min dist: 0.0239120135452
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6, 5, 0]
    found id: 6 ; min dist: 0.431899879288
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6, 5, 0, 6]
    found id: 1 ; min dist: 0.2728550754
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6, 5, 0, 6, 1]
    found id: 2 ; min dist: 0.105319977891
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6, 5, 0, 6, 1, 2]
    found id: 3 ; min dist: 0.323066743806
    id_list [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 6, 5, 0, 6, 1, 2, 3]
    found id: 4 ; min dist: 0.118563870537
    id id_...: found 3 faces
    id id_...: found 3 faces
    id id_...: found 3 faces
    """
    if Config.DEBUG:
        print('RPC-Server debug mode')
        imgs, ids = receive_response_message_link(1)
        detected_face_dict = detect_and_assign(imgs, ids)
        for key, embs_faces in detected_face_dict.items():
            print('id {}: found {} faces'.format(key, len(embs_faces)))
            for (_, img, _, _) in embs_faces:
                show_frame(img, wait_time=1000)

    else:
        print('--RPC SERVER start')
        rb.channel.queue_declare(queue=rb.VINGROUP_RPC_SERVER, exclusive=False)
        rb.channel.basic_consume(on_server_rx_rpc_request, queue=rb.VINGROUP_RPC_SERVER)
        rb.channel.start_consuming()
        print('--Start listening to message')


def receive_response_message_link(msg):
    if Config.DEBUG:
        test_img_dir = r'../data/rpc_server'
        imgs = [misc.imread(img_path) for img_path in glob.glob(os.path.join(test_img_dir, '*'))]
        ids = ['id_1', 'id_2', 'id_3']
    else:
        msg_body = msg.decode('ascii').split(Config.Rabbit.INTER_SEP)
        img_urls = msg_body[0].split(Config.Rabbit.INTRA_SEP)
        ids = msg_body[1].split(Config.Rabbit.INTRA_SEP)
        imgs = []
        if img_urls and ids:
            # only get 1st image
            img_url = img_urls[0]
            print(img_url)
            img_abs_path = os.path.join(Config.RPCServer.DST_PATH, img_url)
            img = misc.imread(img_abs_path)
            imgs.append(img)
        else:
            print('Can not get images or ids list')
    return imgs, ids


def detect_and_assign(imgs, ids):
    embedding_dict = {}
    for img_index, img in enumerate(imgs):
        bboxes, points = detector.detect_face(img)
        embedding_dict[img_index] = []
        for bbox in bboxes:
            face_img = CropperUtils.crop_face(img, bbox)
            face_img = preprocess.whitening(face_img)
            display_face_img, padded_bbox_str = CropperUtils.crop_display_face(img, bbox)
            emb = feature_extractor.extract_features(face_img)
            embedding_dict[img_index].append((emb, display_face_img, padded_bbox_str, bbox))

    id_dict = {}
    for img_index, embs_faces in embedding_dict.items():
        if img_index == 0:
            # not use only top biggest faces here because there maybe big face picture on shirt
            embs_array = np.zeros((len(embs_faces), Config.Matcher.EMB_LENGTH))
            id_list = [i for i in range(len(embs_faces))]
            for face_index, (emb, display_face_img, padded_bbox_str, bbox) in enumerate(embs_faces):
                id_dict[face_index] = [(emb, display_face_img, padded_bbox_str, get_area(bbox))]
                embs_array[face_index] = emb
        else:
            visited = []
            for (emb, display_face_img, padded_bbox_str, bbox) in embs_faces:
                dists = np.sum(np.square(np.subtract(embs_array, emb)), 1)
                min_index = np.argmin(dists)
                found_id = id_list[min_index]
                if Config.DEBUG:
                    print('id_list', id_list)
                    print('found id:', found_id, '; min dist:', dists[min_index])
                    show_frame(display_face_img, wait_time=1000)
                if dists[min_index] > Config.MIN_ASSIGN_THRESHOLD:
                    found_id = len(id_dict)
                    id_dict[found_id] = []
                elif found_id in visited:
                    continue
                id_dict[found_id].append((emb, display_face_img, padded_bbox_str, get_area(bbox)))
                embs_array = np.append(embs_array, [emb], axis=0)
                id_list.append(found_id)
                visited.append(found_id)

    nrof_ids = len(ids)
    nrof_imgs = len(imgs)
    # remove id that dont have enough number of face
    id_dict_keys = list(id_dict.keys())
    for key in id_dict_keys:
        if len(id_dict[key]) < nrof_imgs:
            id_dict.pop(key, None)
    # get top biggest bboxes
    bbox_area_dict = dict([(k, id_dict[k][0][-1]) for k in id_dict.keys()])
    sorted_bb_areas = sorted(bbox_area_dict.items(), key=operator.itemgetter(1), reverse=True)
    arg_sorted_bb_areas = [key for (key, _) in sorted_bb_areas]
    # check all
    if len(id_dict.keys()) < nrof_ids:
        key_dict = dict([(k, v) for k, v in zip(id_dict.keys(), ids[:len(id_dict.keys())])])
        id_dict = dict([(key_dict[k], v) for k, v in id_dict.items()])
    elif len(id_dict.keys()) > nrof_ids:
        # only get top nrof_ids biggest bounding box
        valid_keys = arg_sorted_bb_areas[:nrof_ids]
        key_dict = dict([(k, v) for k, v in zip(valid_keys, ids)])
        id_dict = dict([(key_dict[k], v) for k, v in id_dict.items() if k in valid_keys])
    else:
        key_dict = dict([(k, v) for k, v in zip(id_dict.keys(), ids)])
        id_dict = dict([(key_dict[k], v) for k, v in id_dict.items()])

    return id_dict


def prepare_response_message_link(id_dict):
    msg_list = []
    for face_id, embs_faces in id_dict.items():
        this_id_img_paths = [str(face_id)]
        print('saving', face_id)
        for emb, display_face_img, padded_bbox_str, _ in embs_faces:
            file_name = '{}.{}'.format(face_id, Config.IMG_EXT)
            file_relative_path = os.path.join(Config.RPCServer.SAVE_PATH, file_name)
            file_absolute_path = os.path.join(Config.RPCServer.DST_PATH, file_relative_path)
            misc.imsave(file_absolute_path, display_face_img)
            this_id_img_paths.append(file_relative_path)
        msg_list.append(Config.Rabbit.INTRA_SEP.join(this_id_img_paths))
    msg = Config.Rabbit.INTRA_SEP.join(msg_list)
    return msg


def on_server_rx_rpc_request(ch, method_frame, properties, body):
    print('--RPC Server got request')
    start_time = time.time()
    imgs, ids = receive_response_message_link(body)
    detected_face_dict = detect_and_assign(imgs, ids)
    # print(ids)
    # send to vingroup check-in
    msg = prepare_response_message_link(detected_face_dict)
    ch.basic_publish('', routing_key=properties.reply_to, body=msg)
    ch.basic_ack(delivery_tag=method_frame.delivery_tag)
    # send to central worker
    rb.send_response_from_rpc_to_central(rb.API_SERVER_TO_CENTRAL,
                                         detected_face_dict, 'CHECKIN-AREA')
    print('process time:', time.time() - start_time)


if __name__ == '__main__':
    main()
