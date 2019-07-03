from config import Config
from utils import PickleUtils
import unittest
import os
import glob
from scipy import misc
from matcher import KdTreeMatcher
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from config import ROOT
from preprocess import Preprocessor
import time


INPUT_DIR_MARGIN_0 = '%s/data/matching/matcher_test/margin_0' % ROOT
REGISTER_BASE_DIR = '%s/data/matching/matcher_test/register' % ROOT
EXTRACTOR = FacenetExtractor(FaceGraph())
PROCESSOR = Preprocessor()
DELAY = 0 if Config.DEBUG else 10


class KdTreeMatcherTest(unittest.TestCase):
    '''
    Test matcher with small dataset
    '''

    def setUp(self):
        if not Config.DEBUG:
            raise('To run RabbitMQTest, DEBUG Mode must be on')
        self.matcher = KdTreeMatcher()

    def test_setup_matcher_tup(self):
        '''
        Make sure that matcher tuple is build
        '''
        self.assertTrue(os.path.exists(Config.MATCHER_TUP_FILE))
        self.assertIsNotNone(self.matcher._KdTreeMatcher__matcher_tup)

    def test_registered_faces(self):
        for name, embs in EMBS_DICT.items():
            self.assertEqual(name, self.matcher.match(embs[1], 2)[0])
            self.assertEqual(name, self.matcher.match(embs[2], 2)[0])

    def test_new_faces(self):
        for emb in NEW_EMBS:
            self.assertEqual(Config.Matcher.NEW_FACE, self.matcher.match(emb, 2)[0])
        register(INPUT_DIR_MARGIN_0, 'new')
        time.sleep(Config.Matcher.INDEXING_INTERVAL)
        for emb in NEW_EMBS:
            self.assertEqual('new', self.matcher.match(emb, 2))

    def test_unknow_faces(self):
        ids = [self.matcher.match(emb, 2)[0] for emb in UNKNOW_EMBS]
        self.assertTrue(check_items_equal)
        self.assertEqual(ids[0], Config.Matcher.NEW_FACE)

    def test_number_of_top_matches(self):
        number_of_top_matches = 2
        for emb in UNKNOW_EMBS:
            id_, top_ids = self.matcher.match(emb, number_of_top_matches)
            self.assertEqual(len(top_ids), number_of_top_matches)


class KdTreeMatcherWithBaseRegisterTest(KdTreeMatcherTest):
    '''
    Run all KdTreeMatcherTest with more register image
    '''
    @classmethod
    def setUpClass(cls):
        setup_base_register_face()


def get_unknow_embs():
    return get_known_embs('unknown')


def get_new_embs():
    return get_known_embs('new')


def get_known_embs(name):
    path = os.path.join(INPUT_DIR_MARGIN_0, name)
    image_files = glob.glob(os.path.join(path, '*.jpg'))
    embs = [extract_emb(image_file)[1] for image_file in image_files]
    return embs


def check_items_equal(lst, name):
    print(name, lst)
    return lst[1:] == lst[:-1]


def setup_base_register_face():
    for name in os.listdir(REGISTER_BASE_DIR):
        if os.path.isfile(os.path.join(REGISTER_BASE_DIR, name)):
            continue
        register(REGISTER_BASE_DIR, name)


def setup_register_face():
    '''
    Read first image in each directory and register
    '''
    for name in os.listdir(INPUT_DIR_MARGIN_0):
        if os.path.isfile(os.path.join(INPUT_DIR_MARGIN_0, name)):
            continue
        if name == 'unknown' or name == 'new':
            continue
        register(INPUT_DIR_MARGIN_0, name)


def register(base_dir, name):
    '''
    Register the first face for name directory
    '''
    reg_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE, default={})
    dir_ = os.path.join(base_dir, name)
    register_image_file = os.path.join(dir_, '0.jpg')
    img, emb = extract_emb(register_image_file)
    image_id = name + '_0'
    face_id = name
    reg_dict[image_id] = face_id
    image_pkl_file = os.path.join(Config.LIVE_DIR, '%s.pkl' % image_id)
    PickleUtils.save_pickle(image_pkl_file, value=(img, emb))
    PickleUtils.save_pickle(Config.REG_IMAGE_FACE_DICT_FILE, value=reg_dict)


def clear_database():
    try:
        os.remove(Config.REG_IMAGE_FACE_DICT_FILE)
        for file_ in os.listdir(Config.LIVE_DIR):
            os.remove(os.path.join(Config.LIVE_DIR, file_))
        os.remove(Config.MATCHER_TUP_FILE)
    except Exception:
        pass


def extract_emb(image_file):
    image = misc.imread(image_file)
    processed = PROCESSOR.process(image)
    emb = EXTRACTOR.extract_features(processed)
    return image, emb


if __name__ == '__main__':
    print('This test will alter the database, so only run this locally')
    time.sleep(DELAY)
    Config.Matcher.INDEXING_INTERVAL = 4
    clear_database()
    setup_register_face()
    EMBS_DICT = {
        'binh': get_known_embs('binh'),
        'dat': get_known_embs('dat'),
        'man': get_known_embs('man'),
        'phuc': get_known_embs('phuc')
    }
    NEW_EMBS = get_new_embs()
    UNKNOW_EMBS = get_unknow_embs()
    unittest.main()
