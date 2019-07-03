from config import Config
import re
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
# import tensorflow.contrib.tensorrt as trt
import numpy as np


class FacenetExtractor(object):
    '''
    Using facenet to extract 128 feature vectors
    '''

    def __init__(self,
                 face_rec_graph,
                 model_path=Config.FACENET_DIR):
        '''
        :param face_rec_sess: FaceRecSession object
        :param model_path: path to trained model
        '''
        print("Loading model...")

        with face_rec_graph.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=Config.GPU_FRACTION)
            gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                         log_device_placement=False,
                                                         allow_soft_placement=True))
            with self.sess.as_default():
                with tf.device(Config.GPU_DEVICE):
                    self.__load_model(model_path)
                    self.images_placeholder = tf.get_default_graph() \
                                                .get_tensor_by_name("input:0")
                    self.embeddings = tf.get_default_graph() \
                                        .get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = tf.get_default_graph() \
                                                     .get_tensor_by_name("phase_train:0")
                    self.embedding_size = self.embeddings.get_shape()[1]
                    try:
                        self.coefficients = tf.get_default_graph()\
                                                .get_tensor_by_name("coefficients:0")
                    except KeyError:
                        self.coefficients = tf.constant([[100]])

    def extract_features(self, face_img):
        '''
        Extract 128 feature vector
        :param face_img: 160x160 face
        :return emb: list of embedding vectorinput face
        '''
        tensor = tensorization(face_img)
        feed_dict = {self.images_placeholder: tensor, self.phase_train_placeholder: False}
        emb, coefficient = self.sess.run([self.embeddings, self.coefficients], feed_dict=feed_dict)
        emb = np.squeeze(emb)
        return emb, coefficient

    def extract_features_all_at_once(self, face_imgs):
        '''
        Extract 128 feature vector
        :param face_img: 160x160 face
        :return emb: list of embedding vectorinput face
        '''
        feed_dict = {self.images_placeholder: face_imgs, self.phase_train_placeholder: False}
        emb_array, coeff_array = self.sess.run([self.embeddings, self.coefficients], feed_dict=feed_dict)
        return emb_array, coeff_array

    def __load_model(self, model):
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if os.path.isfile(model_exp):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as file_:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file_.read())
                tf.import_graph_def(graph_def, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = get_model_filenames(model_exp)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file \
                                    in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for file_ in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', file_)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def tensorization(img):
    '''
    Prepare the imgs before input into model
    :param img: Single face image
    :return tensor: numpy array in shape(n, 160, 160, 3) ready for input to cnn
    '''
    tensor = img.reshape(-1, Config.Align.IMAGE_SIZE, Config.Align.IMAGE_SIZE, 3)
    return tensor
