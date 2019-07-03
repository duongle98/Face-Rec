import glob
import argparse
import os
from utils import refresh_folder
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--id',
                    help='the identity u want to search',
                    required=True,
                    default=None)
parser.add_argument('-s',
                    '--search_folder',
                    help='search folder, eg: ../data/tracking',
                    default='../data/tracking')
parser.add_argument('-r',
                    '--result_folder',
                    help='result folder, eg: ../data/search_result',
                    default='../data/search_result')
parser.add_argument('--include_matching',
                    help='results also contain matching threshold images',
                    action='store_true')
args = parser.parse_args()
tracking_dir = args.search_folder
if not os.path.exists(args.result_folder):
    os.mkdir(args.result_folder)
out_dir = os.path.join(args.result_folder, args.id)
refresh_folder(out_dir)
tracker_paths = glob.glob(tracking_dir + '/**')
for tracker_path in tracker_paths:
    img_ids = glob.glob(tracker_path + '/*.jpg')
    folder_name = tracker_path.split('/')[-1]
    for img_id in img_ids:
        img_name = img_id.split('/')[-1]
        if args.id in img_name:
            print('Found: {}'.format(img_id))
            if 'match' in folder_name and args.include_matching:
                if not os.path.exists(os.path.join(out_dir, folder_name)):
                    os.mkdir(os.path.join(out_dir, folder_name))
                copyfile(img_id, os.path.join(out_dir, folder_name, img_name))
            elif 'match' not in folder_name:
                if not os.path.exists(os.path.join(out_dir, folder_name)):
                    os.mkdir(os.path.join(out_dir, folder_name))
                copyfile(img_id, os.path.join(out_dir, folder_name, img_name))
