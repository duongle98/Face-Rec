

import os
import cv2
import pickle
import numpy as np
from math import sqrt


def get_new_key(query_image_id, frame_image_dict):
    # find frame id
    for frame_id, face_list in frame_image_dict.items():
        for image_id, bbox in face_list:
            if query_image_id == image_id:
                bbox_str = '_'.join(np.array(bbox, dtype=np.unicode).tolist())
                new_key = str(frame_id) + '_' + bbox_str
                return new_key
    return None


def show(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('', img)
    cv2.waitKey()


def get_key_by_value(_value, _dict):
    for k, v in _dict.items():
        if v == _value:
            return k


def get_frame_id_from_frame_dict(query_image_id, _dict):
    for key, value_list in _dict.items():
        for (image_id, _) in value_list:
            if query_image_id == image_id:
                return key
    return None


def get_track_frame_id(image_id):
    # image id sample: Office-Server_2_1516076427.0_46_50_-45_-101
    basename = os.path.basename(image_id)
    frame_id = basename.split('_')[1]
    if str.isdigit(frame_id):
        return int(frame_id)
    return 100000


def chop_extension(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def get_emb(image_id, live_dir):
    pickle_file = os.path.join(live_dir, image_id + '.pkl')
    live_tup = pickle.load(open(pickle_file, 'rb'))
    return live_tup[1]


def get_image(image_id, live_dir):
    pickle_file = os.path.join(live_dir, image_id + '.pkl')
    live_tup = pickle.load(open(pickle_file, 'rb'))
    return live_tup[0]


def l2_distance(emb1, emb2):
    return sqrt(np.sum(np.square(np.subtract(emb1, emb2))))


def sum_square_distance(emb1, emb2):
    return np.sum(np.square(np.subtract(emb1, emb2)))


def find_image_id_from_frames_dict(input_string, frames_dict):
    frame_id = get_track_frame_id(input_string)
    # chop bounding box string
    input_bbox_str = '_'.join(input_string.split('_')[-4:])
    if frame_id in frames_dict:
        for tup in frames_dict[frame_id]:
            gt_bbox_str = '_'.join(np.array(tup[1], dtype=np.unicode).tolist())
            if input_bbox_str == gt_bbox_str:
                return tup[0]
    return None


def get_parent_dir_name(path):
    return os.path.basename(os.path.dirname(path))


def chop_image_id_element(_string, start_idx, end_idx=None):
    return '_'.join(_string.split('_')[start_idx:end_idx])


def calc_iou(bb1, bb2):
    """
    Check overlap between 2 bounding boxes
    Return: float in [0,1]
    >>> check_overlap([1, 1, 3, 4], [1, 2, 3, 5])
    0.5
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class Features:
    def __init__(self, embs_ids_file):
        if not os.path.exists(embs_ids_file):
            raise Exception('Can''t locate {}'.format(embs_ids_file))
        with open(embs_ids_file, 'rb') as f:
            data = pickle.load(f)
        self.embs = data['embs']
        self.ids = data['labels']

    def get_emb(self, idx):
        return self.embs[idx]

    def get_id(self, idx):
        return self.ids[idx]

    def get_nrof_embs(self):
        return len(self.ids)
