import motmetrics as mm
import sys
import argparse
import os
import glob


"""
Install motmetrics before running this script

Usage:
python mot_eval.py --gt ground_truth_dir --ht hypothesis_dir
"""


def get_basename(filename):
    return filename[:-3]


def get_parent_dir_name(path):
    return os.path.basename(os.path.dirname(path))


def get_filename(file_path):
    _, filename = os.path.split(file_path)
    return filename


def get_gt_info(file_path):
    filename = get_filename(file_path)
    parent_name = get_parent_dir_name(file_path)
    basename = get_basename(filename)
    basename = basename.split('_')
    original_bb = [int(i) for i in basename[-9:-5]]
    x = original_bb[0]
    y = original_bb[1]
    w = abs(original_bb[2] - original_bb[0])
    h = abs(original_bb[3] - original_bb[1])
    frame_id = int(basename[-5])
    return {
        'track_id': parent_name + "gt",
        'frame_id': frame_id,
        'bb': (x, y, w, h)
    }


def get_hypothesis_info(file_path):
    filename = get_filename(file_path)
    basename = get_basename(filename)
    basename = basename.split('_')
    original_bb = [int(i) for i in basename[1:5]]
    tracker_info = basename[0].split('-')
    if not tracker_info[1] == 'eval':
        return None
    track_id = tracker_info[2]
    x = original_bb[0]
    y = original_bb[1]
    w = abs(original_bb[2] - original_bb[0])
    h = abs(original_bb[3] - original_bb[1])
    frame_id = int(basename[5])
    return {
        'track_id': track_id,
        'frame_id': frame_id,
        'bb': (x, y, w, h)
    }


def generate_from_track_dir(folder_path, get_info):
    query = os.path.join(folder_path, "**", "*")
    data = {}
    for file_path in glob.glob(query):
        file_info = get_info(file_path)
        if file_info:
            frame_id = file_info['frame_id']
            tracker_id = file_info['track_id']
            bb = file_info['bb']
            if frame_id not in data:
                data[frame_id] = [(tracker_id, bb)]
            else:
                data[frame_id].append((tracker_id, bb))
    return data


def print_result(gt, ht):
    interested_frame = set(gt.keys()).union(set(ht.keys()))
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in list(interested_frame):
        gt_ids = []
        gt_bbs = []
        ht_ids = []
        ht_bbs = []
        if frame_id in gt:
            for tracker in gt[frame_id]:
                gt_ids.append(tracker[0])
                gt_bbs.append(tracker[1])

        if frame_id in ht:
            for tracker in ht[frame_id]:
                ht_ids.append(tracker[0])
                ht_bbs.append(tracker[1])
        distance_matrix = mm.distances.iou_matrix(gt_bbs, ht_bbs, max_iou=0.8)
        acc.update(gt_ids, ht_ids, distance_matrix)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc],
        metrics=mm.metrics.motchallenge_metrics,
        names=['Value'])

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


def main(args):
    ht_folder = args.ht
    gt_folder = args.gt
    ground_truth = generate_from_track_dir(gt_folder, get_gt_info)
    hypothesis = generate_from_track_dir(ht_folder, get_hypothesis_info)
    print_result(ground_truth, hypothesis)


def parse_arguments(argv):
    parser = argparse.ArgumentParser('parser',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt', help='path to ground truth folder')
    parser.add_argument('--ht', help='path to hypothesis folder')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
