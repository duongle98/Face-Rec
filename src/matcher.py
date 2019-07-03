from abc import ABCMeta, abstractmethod
import os
import time
import sys
from collections import namedtuple
import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from config import Config
from utils import PickleUtils


KdTreeTuple = namedtuple('KdTreeTuple', ['embs', 'labels', 'length'])
FaissTuple = namedtuple('FaissTuple', ['embs', 'labels', 'length'])
SvmTuple = namedtuple('SvmTuple', ['last_time', 'svm', 'length'])
# LinearTuple = namedtuple('LinearTuple', ['last_time', 'embs', 'length', 'image_ids'])


class AbstractMatcher(metaclass=ABCMeta):

    def __init__(self,
                 threshold=Config.Matcher.MIN_ASSIGN_THRESHOLD):
        '''
        :param threhold: matching threshold for this matcher
        :param indexing_time: how often this matcher will update itself
        '''
        self._threshold = threshold
        self._matcher_tup = None
        self._classifier = None
        print("Abstract matcher")

    def load_model(self, file_name):
        # load matcher from pkl file
        self._matcher_tup = PickleUtils.read_pickle(file_name, default=None)
        if self._matcher_tup is not None:
            self.fit(self._matcher_tup.embs, self._matcher_tup.labels)

    def save_model(self, file_name):
        # save matcher to pkl file
        PickleUtils.save_pickle(file_name, value=self._matcher_tup)

    @abstractmethod
    def fit(self, embs, labels):
        '''
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        pass

    @abstractmethod
    def update(self, new_embs, new_labels):
        '''
        add new embs and labels to current matcher
        :param new_embs: list of embs
        :param new_labels: list of label (face id) for each emb
        '''
        pass

    def build(self, mongodb_faceinfo, use_image_id=False):
        '''
        build matcher using register file
        :param register_file: {image_id: face_id}
        '''
        # reg_image_face_dict = PickleUtils.read_pickle(register_file, default={})
        reg_image_face_dict = mongodb_faceinfo.find({'is_registered': True}, projection={'_id': False})

        # image_ids = list(reg_image_face_dict.keys())
        embs = []
        labels = []

        # for image_id in image_ids:
        #     try:
        #         emb = PickleUtils.read_pickle(
        #             os.path.join(Config.LIVE_DIR, '{}.{}'.format(image_id, Config.PKL_EXT)),)[1]
        #     except TypeError:
        #         with open('../data/fail_read_live_pkl.txt', 'a') as f:
        #             f.write(os.path.join(Config.LIVE_DIR,
        #                                  '%s.%s' % (image_id, Config.PKL_EXT)) + '\n')
        #         continue
        #     embs.append(emb)
        #     if use_image_id:
        #         labels.append(image_id)
        #     else:
        #         labels.append(reg_image_face_dict[image_id])

        for cursor in reg_image_face_dict:
            embs.append(np.array(cursor['embedding']))
            if use_image_id:
                labels.append(cursor['image_id'])
            else:
                labels.append(cursor['face_id'])
        self.fit(embs, labels)

    @abstractmethod
    def match(self, emb, top_matches, return_min_dist=False):
        '''
        Find the neart id the the embedding
        :param emb: 128x1 embedding vector
        :param top_matches: number of closest match to emb
        :param min_dist: should return min distance
        :return track_id: closest id to emb
        :return top_match_ids: X top matches face
        :return dist
        '''
        pass


class KdTreeMatcher(AbstractMatcher):
    '''
    Find neareast id for a embedding by using kd-tree
    '''

    def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold = None, return_dists=False, always_return_closest=True):
        '''
        See superclass doc
        '''
        if threshold is None:
            threshold = self._threshold
        if self._classifier is not None:
            top_matches = min(top_matches, self._matcher_tup.length - 1)
            dists, inds = self._classifier.query(emb, k=top_matches)
            dists = np.squeeze(dists).tolist()
            inds = np.squeeze(inds).tolist()
            predict_id = self._matcher_tup.labels[inds[0]]
            min_dist = dists[0]
            # if dist > self._threshold:
            #     predict_id = Config.Matcher.NEW_FACE
            if min_dist <= threshold:
                top_match_ids = [self._matcher_tup.labels[idx] for (i, idx) in enumerate(inds) if dists[i] <= threshold]
            else:
                top_match_ids = [predict_id]
            dists = dists[0:len(top_match_ids)]
        else:
            top_match_ids = [Config.Matcher.NEW_FACE]
            dists = [-1]
            predict_id = Config.Matcher.NEW_FACE
            min_dist = -1

        if return_dists:
            return top_match_ids, dists
        return top_match_ids

    def fit(self, embs, labels):
        '''
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        length = len(embs)
        print('Fit', length)
        if length > 0:
            reg_mat = np.asarray(embs).reshape((length, Config.Matcher.EMB_LENGTH))
            self._classifier = neighbors.KDTree(reg_mat, leaf_size=Config.Matcher.INDEX_LEAF_SIZE,
                                                metric='euclidean')
            self._matcher_tup = KdTreeTuple(embs, labels, length)
        else:
            self._matcher_tup = None
            self._classifier = None

    def update(self, new_embs, new_labels):
        '''
        add new embs and labels to current matcher
        :param new_embs: list of embs
        :param new_labels: list of label (face id) for each emb
        '''
        if self._matcher_tup is None:
            self.fit(new_embs, new_labels)
        else:
            old_embs = self._matcher_tup.embs
            old_labels = self._matcher_tup.labels
            embs = old_embs + new_embs
            labels = old_labels + new_labels
            # fit the KDTree matcher by rebuilding new one
            self.fit(embs, labels)


class FaissMatcher(AbstractMatcher):
    '''
    Classify face id using Faiss
    '''

    def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold = None, return_dists=False, always_return_closest=True):
        '''
        See superclass doc
        '''
        if threshold is None:
            threshold = self._threshold
        if self._classifier is not None:
            top_matches = min(top_matches, self._matcher_tup.length - 1)
            dists, inds = self._classifier.search(np.array(emb).astype('float32'),
                                                  k=top_matches)
            dists = np.squeeze(dists).tolist()
            inds = np.squeeze(inds).tolist()
            predict_id = self._matcher_tup.labels[inds[0]]
            min_dist = dists[0]
            # if dist > self._threshold:
            #     predict_id = Config.Matcher.NEW_FACE
            if min_dist <= threshold:
                top_match_ids = [self._matcher_tup.labels[idx] for (i, idx) in enumerate(inds) if dists[i] <= threshold]
            else:
                if always_return_closest:
                    top_match_ids = [predict_id]
                else:
                    top_match_ids = []
            dists = dists[0:len(top_match_ids)]
        else:
            top_match_ids = [Config.Matcher.NEW_FACE]
            dists = [-1]
            predict_id = Config.Matcher.NEW_FACE
            min_dist = -1

        if return_dists:
            return top_match_ids, dists
        return top_match_ids

    def fit(self, embs, labels):
        '''
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        length = len(embs)
        if length > 0:
            # only fit if we have data
            self._classifier = faiss.IndexFlatL2(Config.Matcher.EMB_LENGTH)
            reg_mat = np.asarray(embs).reshape((length, Config.Matcher.EMB_LENGTH))
            self._classifier.add(reg_mat.astype('float32'))
            self._matcher_tup = FaissTuple(embs, labels, length)
        else:
            self._classifier = None
            self._matcher_tup = None

    def update(self, new_embs, new_labels):
        '''
        add new embs and labels to current matcher
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        if self._matcher_tup is None:
            # no matcher yet, call fit instead
            self.fit(new_embs, new_labels)
        else:
            # fit classifier to new embs
            length = len(new_embs)
            reg_mat = np.asarray(new_embs).reshape((length, Config.Matcher.EMB_LENGTH))
            self._classifier.add(reg_mat.astype('float32'))

            # update embs and labels
            old_embs = self._matcher_tup.embs
            old_labels = self._matcher_tup.labels
            embs = old_embs + new_embs
            labels = old_labels + new_labels

            # update matcher
            self._matcher_tup = FaissTuple(embs, labels, len(embs))


class SVMMatcher(AbstractMatcher):
    '''
    Classify face id using SVM
    '''

    def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES, return_min_dist=False):
        '''
        See superclass doc
        '''
        if self._matcher_tup is not None:
            # self._rwlock.reader_acquire()
            top_matches = min(top_matches, self._matcher_tup.length - 1)
            predict_id = self._matcher_tup.svm.predict(emb)[0]

            probs = self._matcher_tup.svm.predict_proba(emb)
            probs = np.squeeze(probs)
            dist = max(probs)
            inds = np.argpartition(probs, top_matches)[:top_matches]
            top_track_ids = self._matcher_tup.svm.classes_[inds]
            if dist < 0.6:
                predict_id = Config.Matcher.NEW_FACE
            # self._rwlock.reader_release()
        else:
            predict_id = Config.Matcher.NEW_FACE
            top_track_ids = []
            dist = -1
            

        if return_min_dist:
            return predict_id, top_track_ids, dist
        return predict_id, top_track_ids

    def fit(self, embs, labels):
        t0 = time.time()
        svm = SVC(kernel='linear', probability=True)
        length = len(embs)
        embs = np.asarray(embs).reshape((length, Config.Matcher.EMB_LENGTH)).astype('float32')
        svm.fit(embs, labels)
        t1 = time.time()
        print('Fit time %f' % (t1 - t0))
        length = len(embs)
        # self._rwlock.writer_acquire()
        self._matcher_tup = SvmTuple(time.time(), svm, length)
        # self._rwlock.writer_release()
        t2 = time.time()
        print('Matcher time %f' % (t2 - t1))
        print('Size svm', sys.getsizeof(self._matcher_tup))
        PickleUtils.save_pickle(Config.SVM_MATCHER_TUP_FILE, value=self._matcher_tup)
        t3 = time.time()
        print('Save time %f' % (t3 - t2))
    
    def update(self, force=False):
        print('this classifiers can be updated')

    # @track_calls
    # def update(self, force=False):
    #     '''
    #     See superclass doc
    #     '''
    #     self._reg_image_face_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE,
    #                                                         default={})
    #     if self._should_update(force):
    #         print('Buid svm matcher')
    #         image_ids = list(self._reg_image_face_dict.keys())
    #         labels = list(self._reg_image_face_dict.values())
    #         embs = [PickleUtils.read_pickle(os.path.join(Config.LIVE_DIR,
    #                                                      '{}.{}'.format(image_id,
    #                                                                     Config.PKL_EXT)))[1]
    #                 for image_id in image_ids]
    #         t0 = time.time()
    #         svm = SVC(kernel='linear', probability=True)
    #         svm.fit(embs, labels)
    #         t1 = time.time()
    #         print('Fit time %f' % (t1 - t0))
    #         length = len(embs)
    #         self._rwlock.writer_acquire()
    #         self._matcher_tup = SvmTuple(time.time(), svm, length)
    #         self._rwlock.writer_release()
    #         t2 = time.time()
    #         print('Matcher time %f' % (t2 - t1))
    #         print('Size svm', sys.getsizeof(self._matcher_tup))
    #         PickleUtils.save_pickle(Config.SVM_MATCHER_TUP_FILE, value=self._matcher_tup)
    #         t3 = time.time()
    #         print('Save time %f' % (t3 - t2))


# class LinearMatcher(AbstractMatcher):
#     '''
#     Find nearest id for a embeddings by calculating distance to all recorded embeddings
#     '''

#     def _setup(self):
#         '''
#         Setup matcher code
#         '''
#         self._matcher_tup = PickleUtils.read_pickle(Config.LINEAR_MATCHER_TUP_FILE)
#         self.update()

#     def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES, return_min_dist=False):
#         '''
#         See superclass doc
#         '''
#         if self._matcher_tup is not None:
#             self._rwlock.reader_acquire()
#             top_matches = min(top_matches, self._matcher_tup.length - 1)
#             dists = DistanceUtils.l2_distance(self._matcher_tup.embs, emb)
#             top_matches = min(self._matcher_tup.length, top_matches)

#             inds = np.argpartition(dists, top_matches)[:top_matches]
#             min_ind = np.argmin(dists)
#             predict_id = self._reg_image_face_dict[self._matcher_tup.image_ids[min_ind]]

#             top_match_ids = []
#             if dists[min_ind] > self._threshold:
#                 predict_id = Config.Matcher.NEW_FACE
#             for ind in inds:
#                 id_ = self._reg_image_face_dict[self._matcher_tup.image_ids[ind]]
#                 top_match_ids.append(id_)
#             dist = dists[min_ind]
#             self._rwlock.reader_release()
#         else:
#             predict_id = Config.Matcher.NEW_FACE
#             top_match_ids = []
#             dist = -1

#         if return_min_dist:
#             return predict_id, top_match_ids, dist
#         return predict_id, top_match_ids

#     @track_calls
#     def update(self, force=False):
#         '''
#         See superclass doc
#         '''

#         embs = []
#         self._reg_image_face_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE,
#                                                             default={})

#         if self._should_update(force):
#             print('Build linear matcher')
#             t0 = time.time()
#             image_ids = list(self._reg_image_face_dict.keys())
#             t1 = time.time()
#             print('image id time %f' % (t1 - t0))
#             embs = [PickleUtils.read_pickle(os.path.join(Config.LIVE_DIR,
#                                                          '{}.{}'.format(image_id,
#                                                                         Config.PKL_EXT)))[1]
#                     for image_id in image_ids]
#             embs = np.vstack((embs))
#             t2 = time.time()
#             print('Embs time %f' % (t2 - t1))
#             self._rwlock.writer_acquire()
#             self._matcher_tup = LinearTuple(time.time(), embs, len(embs), image_ids)
#             self._rwlock.writer_release()
#             t3 = time.time()
#             print('Matcher time %f' % (t3 - t2))
#             print('Size li', sys.getsizeof(self._matcher_tup))
#             PickleUtils.save_pickle(Config.LINEAR_MATCHER_TUP_FILE, value=self._matcher_tup)
#             t4 = time.time()
#             print('Save time %f' % (t4 - t3))
