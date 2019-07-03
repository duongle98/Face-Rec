import pickle
import os
import sys
import shutil
import time


def main(session_dir):
    save_dir = os.path.join(os.getcwd(),
                            '../results',
                            '{}-{}'.format('qa-database', str(time.time())))
    print(save_dir)
    face_changes_list_file = os.path.join(session_dir,
                                          'db',
                                          'face_changes_list.pkl')
    predictions_file = os.path.join(session_dir, 'db', 'predictions.pkl')

    live_dir = os.path.join(session_dir, 'live')
    frames_file = os.path.join(session_dir, 'db', 'frame_image_dict.pkl')
    reg_image_face_dict_file = os.path.join(session_dir,
                                            'db',
                                            'reg_image_face_dict.pkl')
    frames_dir = os.path.join(session_dir, 'frames')

    face_changes_list = pickle.load(open(face_changes_list_file, 'rb'))
    predictions = pickle.load(open(predictions_file, 'rb'))

    print('predictions length: ', len(predictions))
    print('face_changes lenght: ', len(face_changes_list))

    nrof_not_found_image_ID = 0
    for (_, image_ID, fixed_label) in face_changes_list:
        try:
            print(image_ID, predictions[image_ID], '->', fixed_label)
            predictions[image_ID] = fixed_label
        except KeyError:
            print(image_ID)
            nrof_not_found_image_ID += 1
    print('missing image_IDs from predictions:', nrof_not_found_image_ID)

    # write all files
    print('start to save files')
    os.mkdir(save_dir)
    save_predictions = os.path.join(save_dir, 'groundtruth.pkl')
    pickle.dump(predictions, open(save_predictions, 'wb'))
    save_reg_img_file = os.path.join(save_dir, 'reg_image_face_dict.pkl')
    shutil.copy(reg_image_face_dict_file, save_reg_img_file)
    shutil.copy(frames_file, os.path.join(save_dir, 'frames.pkl'))
    shutil.copytree(frames_dir, os.path.join(save_dir, 'frames'))
    shutil.copytree(live_dir, os.path.join(save_dir, 'live'))

    print('result saved at', save_dir)


main(sys.argv[1])
