

import os
import cv2
import sys
import glob
import pickle
import argparse
from qa_utils import get_track_frame_id, get_emb, chop_extension, get_distance, \
                  find_image_id_from_frames_dict, get_frame_id_from_frame_dict, \
                  get_image, get_key_by_value


def main(args):
    qa_dir = args.qa_dir
    trackers_dir = args.trackers_dir
    input_image_id = args.image_id

    live_dir = os.path.join(qa_dir, 'live')
    frames_dict_file = os.path.join(qa_dir, 'frames.pkl')
    frames_dict = pickle.load(open(frames_dict_file, 'rb'))
    ground_truth_file = os.path.join(qa_dir, 'groundtruth.pkl')
    groundtruth = pickle.load(open(ground_truth_file, 'rb'))
    reg_img_file = os.path.join(qa_dir, 'reg_image_face_dict.pkl')
    reg_img_dict = pickle.load(open(reg_img_file, 'rb'))

    # indexing trackers_dir
    trackers_dict = {}
    d = os.path.join(trackers_dir, '*')
    tracker_ids = [os.path.basename(p) for p in glob.glob(d)]
    for tracker_id in tracker_ids:
        d = os.path.join(trackers_dir, tracker_id, '*')
        image_list = [chop_extension(p) for p in glob.glob(d)]
        # find end frame_id of this tracker
        frame_ids = [get_track_frame_id(image_id) for image_id in image_list]
        start_frame = min(frame_ids)
        end_frame = max(frame_ids)
        trackers_dict[tracker_id] = (start_frame, end_frame, image_list)

    # sample image_id: Office-Server_2_1516076427.0_46_50_-45_-101
    current_frame_id = get_frame_id_from_frame_dict(input_image_id, frames_dict)
    current_trackers = []
    for tracker_id, (start_frame, end_frame, _) in trackers_dict.items():
        if current_frame_id > start_frame and current_frame_id < end_frame:
            current_trackers.append(tracker_id)

    input_face_id = groundtruth[input_image_id]
    register_img_id = get_key_by_value(input_face_id, reg_img_dict)
    register_emb = get_emb(register_img_id, live_dir)
    input_emb = get_emb(input_image_id, live_dir)
    dist = get_distance(input_emb, register_emb)
    print('distance to register image: ', dist)
    input_img = cv2.cvtColor(get_image(input_image_id, live_dir), cv2.COLOR_RGB2BGR)
    cv2.imshow(input_image_id, input_img)

    for tracker_id in current_trackers:
        this_tracker_imgs = trackers_dict[tracker_id][2]
        for track_image in this_tracker_imgs:
            if current_frame_id > get_track_frame_id(track_image):
                track_image_id = find_image_id_from_frames_dict(track_image, frames_dict)
                if track_image_id:
                    track_emb = get_emb(track_image_id, live_dir)
                    dist = get_distance(input_emb, track_emb)
                    print(tracker_id, track_image, dist)
                    img_path = os.path.join(trackers_dir, tracker_id, track_image + '.jpg')
                    img = cv2.imread(img_path)
                    cv2.imshow(track_image, img)
                    cv2.waitKey()


def parse_agruments(agrv):
    parser = argparse.ArgumentParser('parses agrument for compare trackers',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--qa_dir', help='QA database dir',
                        default='/media/loi/E/Work/EyeQ/Stuff/tracking-data/door_1_qa-database')
    parser.add_argument('--trackers_dir', help='tracking results dir',
                        default='/media/loi/E/Work/EyeQ/Stuff/tracking')
    parser.add_argument('--image_id', help='query image id',
                        default='Office-Server_2_1516076427.0_46_50_-45_-101')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_agruments(sys.argv[1:]))
