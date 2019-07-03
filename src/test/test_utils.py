import unittest
from config import ROOT
from face_detector import MTCNNDetector
from tf_graph import FaceGraph
from utils import CropperUtils
from scipy import misc
import numpy as np

IMAGES = {
    'big_face': misc.imread('%s/data/cropper/big_face.jpg' % ROOT),
    'top': misc.imread('%s/data/cropper/top.jpg' % ROOT),
    'bottom': misc.imread('%s/data/cropper/bottom.jpg' % ROOT),
    'left': misc.imread('%s/data/cropper/left.jpg' % ROOT),
    'right': misc.imread('%s/data/cropper/right.jpg' % ROOT),
    'out_range_top': misc.imread('%s/data/cropper/out_range_top.jpg' % ROOT),
    'out_range_bottom': misc.imread('%s/data/cropper/out_range_bottom.jpg' % ROOT),
    'out_range_right': misc.imread('%s/data/cropper/out_range_right.jpg' % ROOT),
    'out_range_left': misc.imread('%s/data/cropper/out_range_left.jpg' % ROOT)
}

DETECTOR = MTCNNDetector(FaceGraph())


class CropperUtilsTest(unittest.TestCase):
    '''
    Run testing on cropping, assume that all face is in acceptable angle and is inner of range
    '''

    def test_display_face_ratio(self):
        [self.display_face_ratio(name, image) for name, image in IMAGES.items()]

    def test_revesed_face_same_as_cropped_face(self):
        [self.reverse_face_same_as_cropped_face(name, image) for name, image in IMAGES.items()]

    def test_cropped_shape(self):
        [self.cropped_shape(name, image) for name, image in IMAGES.items()]

    def reverse_face_same_as_cropped_face(self, name, image):
        bbs, pts = DETECTOR.detect_face(image)
        cropped_face = CropperUtils.crop_face(image, bbs[0])
        display_face, padded_bb_str = CropperUtils.crop_display_face(image, bbs[0])
        reversed_face = CropperUtils.reverse_display_face(display_face, padded_bb_str)
        self.assertEqual(cropped_face.shape, reversed_face.shape, msg=name)
        self.assertAlmostEqual(np.sum(cropped_face - reversed_face), 0, msg=name)

    def display_face_ratio(self, name, image):
        bbs, pts = DETECTOR.detect_face(image)
        display_face, _ = CropperUtils.crop_display_face(image, bbs[0])
        h, w, _ = display_face.shape
        self.assertGreater(h, 0, msg=name)
        self.assertGreater(w, 0, msg=name)
        self.assertAlmostEqual(h/w - 1.5, 0, delta=0.01, sg=name)

    def cropped_shape(self, name, image):
        bbs, pts = DETECTOR.detect_face(image)
        cropped_face = CropperUtils.crop_face(image, bbs[0])
        h, w, _ = cropped_face.shape
        self.assertEqual(h, 160, msg=name)
        self.assertEqual(w, 160, msg=name)


if __name__ == '__main__':
    unittest.main()
