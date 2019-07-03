'''
Doc string
'''
import tensorflow as tf


class FaceGraph(object):
    '''
    Doc string
    '''
    def __init__(self, gpu_memory_fraction=0.1):
        '''
            There'll be more to come in this class
        '''
        print(gpu_memory_fraction)
        with tf.device('/device:GPU:1'):
            self.graph = tf.Graph()
