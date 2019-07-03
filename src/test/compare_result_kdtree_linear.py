import os
from matcher import LinearMatcher, KdTreeMatcher
from utils import PickleUtils
from config import Config


def generate_smaller_reg_dict():
    reg_image_face_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE,
                                                  default={})
    ids = set([reg_image_face_dict[fid] for fid in reg_image_face_dict])
    smaller_reg_dict = {}
    for single_id in ids:
        img_ids = [key for key, value in reg_image_face_dict.items() if value == single_id]
        for i in range(5):
            smaller_reg_dict[img_ids[i]] = single_id
    PickleUtils.save_pickle('../session/db/smaller_reg_dict.pkl', value=smaller_reg_dict)
    print('Smaller reg dict created.')


def compare_function():
    kd_m = KdTreeMatcher()
    li_m = LinearMatcher()
    smaller_dict = PickleUtils.read_pickle('../session/db/smaller_reg_dict.pkl',
                                           default={})
    if smaller_dict == {}:
        print('Missing ../session/db/smaller_reg_dict.pkl')
        return -1
    test_img_ids = list(smaller_dict.keys())
    test_embs = [PickleUtils.read_pickle(os.path.join(Config.LIVE_DIR,
                                                      '{}.{}'.format(image_id, Config.PKL_EXT)))[1]
                 for image_id in test_img_ids]
    same = 0
    not_same = 0
    for emb in test_embs:
        kd_id, _ = kd_m.match(emb)
        li_id, _ = li_m.match(emb)
        if kd_id == li_id:
            same += 1
        else:
            not_same += 1

    print('Same: {}\nNot Same: {}'.format(same, not_same))


compare_function()
