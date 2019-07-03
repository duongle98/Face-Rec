import sys
import numpy as np
import os
import argparse
import glob
import cv2
from scipy import misc
from utils import CropperUtils
from preprocess import default_preprocess
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph


def main(args):
    print(args.image_dir)
    if args.image_files is None:
        image_files = glob.glob(os.path.join(args.image_dir, r'*.*'))
    else:
        image_files = args.image_files

    print(image_files)
    face_graph = FaceGraph()
    images = load_and_align_data(face_graph, image_files, args.display)
    extractor = FacenetExtractor(face_graph)

    emb_list = []
    for image in images:
        _image = default_preprocess(image)
        _emb = extractor.extract_features(_image)
        emb_list.append(_emb)

    emb = np.vstack(emb_list)
    nrof_images = len(image_files)

    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, image_files[i]))
    print('')

    # Print distance matrix
    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %1d     ' % i, end='')
    print('')
    for i in range(nrof_images):
        print('%1d  ' % i, end='')
        for j in range(nrof_images):
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
            print('  %1.4f  ' % dist, end='')
        print('')


def load_and_align_data(face_graph, image_paths, do_display):
    detector = MTCNNDetector(face_graph)
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        bounding_boxes, landmarks = detector.detect_face(img)

        all_face_marks_x = landmarks[0:5, :]
        all_face_marks_y = landmarks[5:10, :]

        face_marks_x = all_face_marks_x[:, 0]
        face_marks_y = all_face_marks_y[:, 0]

        print(face_marks_x)
        print(face_marks_y)

        # draw landmarks on image
        if do_display:
            frame = img.copy()
            for x, y in zip(face_marks_x, face_marks_y):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.imshow(image_paths[i], frame)
            cv2.waitKey()
        bounding_box = bounding_boxes[0]
        img_list[i] = CropperUtils.crop_face(img, bounding_box)
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Directory containing images to compare')
    parser.add_argument('--image_files', type=str, default=None, nargs='+',
                        help='Images to compare')
    parser.add_argument('-d', '--display', help='choose to display image', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
