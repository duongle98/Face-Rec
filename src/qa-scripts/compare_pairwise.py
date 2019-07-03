

import os
import sys
import glob
import random
import argparse
import numpy as np
from qa_utils import chop_extension, chop_image_id_element, calc_iou


def get_padded_bbox(image_id):
    coor_list = image_id.split('_')[-4:]
    bbox = np.array((coor_list), dtype=np.int32)
    return bbox


def get_origin_bbox(image_id):
    coor_list = image_id.split('_')[:4]
    bbox = np.array((coor_list), dtype=np.int32)
    return bbox


def find_original_image_id(input_image_id, gt_frame_image_dict, use_origin_bb=False):
    input_frame_id = input_image_id.split('_')[4]
    if use_origin_bb:
        input_bbox = get_origin_bbox(input_image_id)
    else:
        input_bbox = get_padded_bbox(input_image_id)

    if input_frame_id in gt_frame_image_dict:
        for image_id in gt_frame_image_dict[input_frame_id]:
            if use_origin_bb:
                img_bbox = get_origin_bbox(image_id)
            else:
                img_bbox = get_padded_bbox(image_id)
            if calc_iou(input_bbox, img_bbox) > 0.3:
                return image_id
    return None


def main(args):
    groundtruth_dir = args.ground_truth
    test_data_dir = args.test_data

    print('generating groundtruth dict...')
    d = os.path.join(groundtruth_dir, '*')
    gt_label_dirs = glob.glob(d)
    groundtruth_dict = {}
    gt_false_positive = []
    gt_frame_image_dict = {}
    nrof_gt_pos_pairs = 0

    for label_dir in gt_label_dirs:
        label = os.path.basename(label_dir)
        image_paths = glob.glob(os.path.join(label_dir, '*.jpg'))
        nrof_images = len(image_paths)
        nrof_gt_pos_pairs += np.math.factorial(nrof_images) /\
            (np.math.factorial(2) * np.math.factorial(nrof_images - 2))
        for image_path in image_paths:
            image_id = chop_image_id_element(chop_extension(image_path), -9)
            if 'fp' not in label.lower() and 'bad' not in label.lower():
                groundtruth_dict[image_id] = label
                frame_id = image_id.split('_')[4]
                if frame_id in gt_frame_image_dict:
                    gt_frame_image_dict[frame_id].append(image_id)
                else:
                    gt_frame_image_dict[frame_id] = [image_id]
            else:
                gt_false_positive.append(image_id)

    print('generating test data dict...')
    d = os.path.join(test_data_dir, '**', '*.jpg')
    td_image_paths = glob.glob(d, recursive=True)
    test_data_dict = {}
    td_bad_tracks = []
    missing_images = 0

    for image_path in td_image_paths:
        if 'match' not in image_path.lower():
            label = chop_extension(image_path).split('_')[0]
            if args.new_image_id_format:
                _image_id = chop_image_id_element(chop_extension(image_path), -10, -1)
            else:
                _image_id = chop_image_id_element(chop_extension(image_path), -9)
            image_id = find_original_image_id(_image_id, gt_frame_image_dict, args.use_origin_bb)
            if args.verbose:
                print('ori: {}, found: {}'.format(_image_id, image_id))
            if image_id:
                if 'bad' not in image_path.lower():
                    test_data_dict[image_id] = label
                else:
                    td_bad_tracks.append(image_id)
            else:
                missing_images += 1
    test_data_keys = set(test_data_dict.keys())
    interest_images = list(set(groundtruth_dict.keys()).intersection(test_data_keys))
    false_positive_in_test_data = list(set(gt_false_positive).intersection(test_data_keys))
    unknown_images = test_data_keys.difference(set(interest_images).union(
                                               set(false_positive_in_test_data)))
    # generate pairs from the interest_image ~ both appear in groundtruth and test data
    print('\ngenerating pos/neg pairs...')
    pos_pairs = []
    neg_pairs = []
    for i in range(len(interest_images) - 1):
        for j in range(i + 1, len(interest_images)):
            image_i = interest_images[i]
            image_j = interest_images[j]
            if groundtruth_dict[image_i] == groundtruth_dict[image_j]:
                pos_pairs.append((image_i, image_j))
            else:
                neg_pairs.append((image_i, image_j))

    nrof_sample = min(len(pos_pairs), len(neg_pairs))
    neg_pairs = random.sample(neg_pairs, nrof_sample)
    pos_pairs = random.sample(pos_pairs, nrof_sample)

    print('calculating tp, fp...')
    fp = 0
    tp = 0
    for pair in pos_pairs:
        if test_data_dict[pair[0]] == test_data_dict[pair[1]]:
            tp += 1
        else:
            fp += 1

    print('calculating tn, fn...')
    tn = 0
    fn = 0
    for pair in neg_pairs:
        if test_data_dict[pair[0]] == test_data_dict[pair[1]]:
            fn += 1
        else:
            tn += 1

    print('\nResult:-----------------------')
    print('nrof false positive in test: ', len(false_positive_in_test_data))
    print('nrof bad tracks in test: ', len(td_bad_tracks))
    print('nrof not-in-groundtruth images', len(unknown_images) + missing_images)
    print('nrof pairs: ', nrof_sample)
    print('true positive: ', tp)
    print('false positive: ', fp)
    print('true negative: ', tn)
    print('false negative: ', fn)
    print('FAR: {}%'.format(round(fp*100 / (fp + tn + 1e-6), 2)))
    print('FRR: {}%'.format(round(fn*100 / (fn + tp + 1e-6), 2)))
    print('Precision: {}%'.format(round(tp / (tp + fp + 1e-6)*100, 2)))
    print('Recal base on test data: {}%'.format(round(tp*100 / (tp + fn + 1e-6), 2)))
    # recall is calculate by the nrof true prediction over nrof available pos pairs in gt
    print('Recall base on groundtruth: {}%'.format(round(tp*100 / (nrof_gt_pos_pairs + 1e-6), 2)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser('parser for pairwise compare',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ground_truth',
                        '-gt',
                        help='Ground truth folder')
    parser.add_argument('--test_data',
                        '-td',
                        help='new data directory after run algorithm')
    parser.add_argument('--use_origin_bb',
                        '-uob',
                        help='use original bb to generate unique key',
                        action='store_true')
    parser.add_argument('--verbose',
                        '-v',
                        help='print out log lines',
                        action='store_true')
    parser.add_argument('--new_image_id_format',
                        '-nif',
                        help='new image_id format have trackid at the end',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
