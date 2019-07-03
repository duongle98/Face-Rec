import pickle
import os
import sys
from scipy import misc
import shutil
import argparse
import numpy as np
from qa_utils import get_new_key, get_key_by_value


def main(args):
    database_dir = args.database
    predictions_file = args.predictions
    WRITE_MODE = args.write
    DEGUG_MODE = args.debug

    if not os.path.exists(database_dir) or not os.path.exists(predictions_file):
        print('can not find database dir or prediction file')
        return

    live_dir = os.path.join(database_dir, 'live')
    predict_file_name = os.path.basename(predictions_file)
    print(predict_file_name)

    groundtruths_raw = pickle.load(open(os.path.join(database_dir, 'groundtruth.pkl'), 'rb'))
    predictions_raw = pickle.load(open(predictions_file, 'rb'))
    frames_dict = pickle.load(open(os.path.join(database_dir, 'frames.pkl'), 'rb'))

    # save [new_key] = old_key
    key_dict = {}
    groundtruths = {}
    for k, v in groundtruths_raw.items():
        new_key = get_new_key(k, frames_dict)
        groundtruths[new_key] = v.lower()
        key_dict[new_key] = k

    predictions = {}
    for k, v in predictions_raw.items():
        new_key = k
        predictions[new_key] = v.lower()
        if predictions[new_key] == 'non_face':
            predictions[new_key] = 'new-face'

    # print(groundtruths)
    groundtruth_keys = set(groundtruths.keys())
    print('groundtruth_keys:', len(groundtruth_keys))
    predictions_keys = set(predictions.keys())
    print('predictions_keys:', len(predictions_keys))

    groundtruth_non_faces = set(key for key, value in groundtruths.items() if 'non_face' in value)
    print('groundtruth_non_faces:', len(groundtruth_non_faces))
    # get image_id that is faces to measure preformance of recognition
    predictions_faces = predictions_keys.difference(groundtruth_non_faces)
    print('predictions_faces:', len(predictions_faces))

    missing_image_ids = groundtruth_keys.difference(predictions_keys)
    print('missing_image_ids:', len(missing_image_ids))
    new_found_image_ids = predictions_keys.difference(groundtruth_keys)
    print('new_found_image_ids:', len(new_found_image_ids))
    # all face images that both in groundtruth and predict
    available_image_ids = predictions_faces.difference(new_found_image_ids)
    print('number of vailable image ids to compare:', len(available_image_ids))

    # images that remove at the detection step in new algo
    new_true_predict_as_non_face = groundtruth_non_faces.difference(predictions_keys)
    # calc false recognition case
    false_recognitions = []
    for image_id in available_image_ids:
        if predictions[image_id] != groundtruths[image_id]:
            false_recognitions.append(image_id)

    # nrof_non_face_in_predictions = len(predictions_keys) - len(predictions_faces)
    print('new_true_predict_as_non_face:', len(new_true_predict_as_non_face))
    print('nrof_false_recognition:', len(false_recognitions))
    # print('overall precision:', 1-(len(false_recognitions) +
    #                             nrof_non_face_in_predictions)/len(predictions_keys))
    print('recognition precision:', 1-len(false_recognitions)/len(available_image_ids))

    if DEGUG_MODE:
        result_list = []
        input_image_dict = {}
        for image_id in false_recognitions:
            reg_img_file = os.path.join(database_dir, 'reg_image_face_dict.pkl')
            reg_image_face_dict = pickle.load(open(reg_img_file, 'rb'))
            groundtruth_id = groundtruths[image_id]
            predict_id = predictions[image_id]
            origin_image_id = key_dict[image_id]
            groundtruth_image_id = get_key_by_value(groundtruth_id, reg_image_face_dict)
            predict_image_id = get_key_by_value(predict_id, reg_image_face_dict)
            if groundtruth_image_id and predict_image_id:
                origin_pkl_file = os.path.join(live_dir, origin_image_id + '.pkl')
                input_emb = pickle.load(open(origin_pkl_file, 'rb'))[1]

                groundtruth_pkl_file = os.path.join(live_dir, groundtruth_image_id + '.pkl')
                groundtruth_emb = pickle.load(open(groundtruth_pkl_file, 'rb'))[1]

                predict_pkl_file = os.path.join(live_dir, predict_image_id + '.pkl')
                predict_emb = pickle.load(open(predict_pkl_file, 'rb'))[1]

                gt_dist = np.sum(np.square(np.subtract(input_emb, groundtruth_emb)))
                pt_dist = np.sum(np.square(np.subtract(input_emb, predict_emb)))
                print(image_id, groundtruth_id, predict_id, gt_dist, pt_dist)
                result_list.append([origin_image_id, groundtruth_id, predict_id, gt_dist, pt_dist])
            # print(groundtruth_image_id, predict_image_id)
            live_tup = pickle.load(open(os.path.join(live_dir, origin_image_id + '.pkl'), 'rb'))
            input_image_dict[origin_image_id] = live_tup[0]
            # show(live_tup[0])

        predictions_name = os.path.splitext(predict_file_name)[0]
        save_dir = os.path.join('../results', predictions_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'door_1_ver3.txt'), 'w') as f:
            for tup in result_list:
                line = ' '.join(str(e) for e in tup)
                f.write(line + '\n')
        for image_id, image in input_image_dict.items():
            misc.imsave(os.path.join(save_dir, image_id + '.jpg'), image)

    if WRITE_MODE:
        # dig in to false_recognition image_ids
        # One_Person_Multi_IDs, Multi_Person_One_IDs
        One_Person_Multi_IDs = {}
        Multi_Person_One_IDs = {}
        for image_id in false_recognitions:
            groundtruth_id = groundtruths[image_id]
            predict_id = predictions[image_id]

            if groundtruth_id in One_Person_Multi_IDs:
                One_Person_Multi_IDs[groundtruth_id].append((image_id, predict_id))
            else:
                One_Person_Multi_IDs[groundtruth_id] = [(image_id, predict_id)]

            if predict_id in Multi_Person_One_IDs:
                Multi_Person_One_IDs[predict_id].append((image_id, groundtruth_id))
            else:
                Multi_Person_One_IDs[predict_id] = [(image_id, groundtruth_id)]

    # writing to file
        predictions_name = os.path.splitext(predict_file_name)[0]
        save_dir = os.path.join('../results', predictions_name)
        if not os.path.exists('./results'):
            os.mkdir('../results')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        OPMI_file = os.path.join(save_dir, 'One_Person_Multi_IDs.pkl')
        pickle.dump(One_Person_Multi_IDs, open(OPMI_file, 'wb'))
        MPOI_file = os.path.join(save_dir, 'Multi_Person_One_IDs.pkl')
        pickle.dump(Multi_Person_One_IDs, open(MPOI_file, 'wb'))

        One_Person_Multi_IDs_dir = os.path.join(save_dir, 'One_Person_Multi_IDs')
        Multi_Person_One_IDs_dir = os.path.join(save_dir, 'Multi_Person_One_IDs')
        if os.path.exists(One_Person_Multi_IDs_dir):
            shutil.rmtree(One_Person_Multi_IDs_dir)
        if os.path.exists(Multi_Person_One_IDs_dir):
            shutil.rmtree(Multi_Person_One_IDs_dir)
        os.mkdir(One_Person_Multi_IDs_dir)
        os.mkdir(Multi_Person_One_IDs_dir)

        print('--writing One_Person_Multi_IDs images')
        for groundtruth_id, image_id_list in One_Person_Multi_IDs.items():
            id_dir = os.path.join(One_Person_Multi_IDs_dir, groundtruth_id)
            if not os.path.exists(id_dir):
                os.mkdir(id_dir)
            for image_id, predict_id in image_id_list:
                image_id = key_dict[image_id]
                (img, _) = pickle.load(open(os.path.join(live_dir, image_id + '.pkl'), 'rb'))
                img_name = '{}|{}.{}'.format(predict_id, image_id, 'jpg')
                img_save_path = os.path.join(id_dir, img_name)
                misc.imsave(img_save_path, img)

        print('--writing Multi_Person_One_IDs images')
        for predict_id, image_id_list in Multi_Person_One_IDs.items():
            id_dir = os.path.join(Multi_Person_One_IDs_dir, predict_id)
            if not os.path.exists(id_dir):
                os.mkdir(id_dir)
            for image_id, groundtruth_id in image_id_list:
                image_id = key_dict[image_id]
                (img, _) = pickle.load(open(os.path.join(live_dir, image_id + '.pkl'), 'rb'))
                img_name = '{}|{}.{}'.format(groundtruth_id, image_id, 'jpg')
                img_save_path = os.path.join(id_dir, img_name)
                misc.imsave(img_save_path, img)

    # for key in One_Person_Multi_IDs.keys():
    #     print(key, [id for _,id in One_Person_Multi_IDs[key]])
    # print('SASDSADS')
    # for key in Multi_Person_One_IDs.keys():
    #     print(key, [id for _,id in Multi_Person_One_IDs[key]])

    # if not os.path.exists('results'):
    #     os.mkdir('results')
    # save_file_name = predict_file_name[:-4] + '_results.pkl'
    # pickle.dump((One_Person_Multi_IDs, Multi_Person_One_IDs),
    #             open(os.path.join('results',save_file_name),'wb'))


def parse_agruments(argv):
    parser = argparse.ArgumentParser('Parses agrument for compare script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data', '--database', help='QA database dir')
    parser.add_argument('-p', '--predictions', help='predictions file path')
    parser.add_argument('-w', '--write', help='write files or not', action='store_true')
    parser.add_argument('-d', '--debug', help='write files or not', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_agruments(sys.argv[1:]))
