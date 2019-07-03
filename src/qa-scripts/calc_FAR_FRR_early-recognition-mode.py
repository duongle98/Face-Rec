
import os
import sys
import time
import random
import pickle
import argparse
import itertools
import numpy as np
from qa_config import Config
from qa_utils import l2_distance, sum_square_distance, chop_extension
from qa_classes import Features, IdentityManager
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from plot import print_metrics, print_distance_histogram, TSNE_visualize
from collections import OrderedDict


class Evaluation:
        # def __init__(self, embs, ids):
        #     self.embs = embs
        #     self.ids = ids

    def load_data(self, embs_ids_file, model_2_data=None):
        self.features = Features(embs_ids_file)
        self.embs_ids_file = chop_extension(embs_ids_file)
        if model_2_data:
            self.model_2_features = Features(model_2_data)

    def generate_identity_manager(self, limit_nrof_faces=None):
        identity_manager_dict = {}
        # group all embs to its id
        for idx in range(self.features.get_nrof_embs()):
            label = self.features.get_id(idx)
            tracker_id = self.features.get_tracker_id(idx)
            if label not in identity_manager_dict:
                identity_manager_dict[label] = IdentityManager()
            identity_manager_dict[label].add_idx(tracker_id, idx)

        # find 1st tracker
        for _, identity_manager in identity_manager_dict.items():
            identity_manager.update_curret_1st_tracker_id()
            identity_manager.sort_idxs_by_frame_id(self.features)
        # sort dictionary buy nrof faces of each id
        return_dict = OrderedDict()
        sorted_items = sorted(identity_manager_dict.items(), key=lambda x: x[1].get_nrof_idxs(), reverse=True)
        for k, v in sorted_items:
            return_dict[k] = v
        return return_dict

    def generate_compare_pairs(self, identity_manager_dict, nrof_pairs, limit_nrof_faces):
        pos_pairs = []
        neg_pairs = []

        identities = list(identity_manager_dict.keys())
        for i, identity in enumerate(identities):
            identity_manager = identity_manager_dict[identity]
            # generate this identity positive pairs
            this_identity_pos_pairs = []
            this_identity_neg_pairs = []
            first_tracker = identity_manager.get_1st_tracker()
            other_trackers = identity_manager.get_other_trackers_list(nrof_sample=-2)

            # get negative pair generator pool
            neg_sample_pool = []
            for other_identity in identities[i + 1:]:
                neg_sample_pool += identity_manager_dict[other_identity].get_1st_tracker()
            for tracker in other_trackers:
                tracker_pos_pairs = list(itertools.product(tracker, first_tracker))
                this_identity_pos_pairs += tracker_pos_pairs

                tracker_neg_pairs = []
                for i in range(len(tracker_pos_pairs)):
                    tracker_neg_pairs.append((random.choice(tracker),
                                              random.choice(neg_sample_pool)))
                this_identity_neg_pairs += tracker_neg_pairs

            pos_pairs += this_identity_pos_pairs
            neg_pairs += this_identity_neg_pairs

        print(len(pos_pairs), len(neg_pairs))
        min_nrof_pairs = min(len(pos_pairs), len(neg_pairs))
        nrof_pairs = min(nrof_pairs, min_nrof_pairs)
        return_pos_pairs = random.sample(pos_pairs, nrof_pairs)
        return_neg_pairs = random.sample(neg_pairs, nrof_pairs)

        print('number of pair: ', nrof_pairs)
        return return_pos_pairs, return_neg_pairs

    def get_dist_actual_issame_array(self, pos_pairs, neg_pairs, distance_type='l2'):
        distances = []
        actual_issame = []
        for (emb_idx_1, emb_idx_2) in pos_pairs:
            emb_1 = self.features.get_emb(emb_idx_1)
            emb_2 = self.features.get_emb(emb_idx_2)
            dist = self.__get_distance(emb_1, emb_2, distance_type)
            distances.append(dist)
            actual_issame.append(True)
            if self.features.get_id(emb_idx_1) != self.features.get_id(emb_idx_2):
                print(self.features.get_id(emb_idx_1), self.features.get_id(emb_idx_2), dist)
        print(sum(distances))

        for (emb_idx_1, emb_idx_2) in neg_pairs:
            emb_1 = self.features.get_emb(emb_idx_1)
            emb_2 = self.features.get_emb(emb_idx_2)
            dist = self.__get_distance(emb_1, emb_2, distance_type)
            distances.append(dist)
            actual_issame.append(False)
            if self.features.get_id(emb_idx_1) == self.features.get_id(emb_idx_2):
                print(self.features.get_id(emb_idx_1), self.features.get_id(emb_idx_2), dist)
        print(sum(distances))

        return np.array(distances), np.array(actual_issame)

    def calculate_metrics(self, distances, actual_issame):
        f1_scores = []
        accuracy_scores = []
        precisions = []
        recalls = []

        min_distance = round(distances.min(), 2)
        max_distance = round(distances.max(), 2)
        print(min_distance, max_distance)
        thresholds = np.arange(min_distance + 0.01, max_distance)# max_distance - 0.01, 0.01)
        # print(len(thresholds))
        for threshold in thresholds:
            # start = time.time()
            predict_issame = distances < threshold
            f1_scores.append(f1_score(actual_issame, predict_issame))
            accuracy_scores.append(accuracy_score(actual_issame, predict_issame))
            precisions.append(precision_score(actual_issame, predict_issame))
            recalls.append(recall_score(actual_issame, predict_issame))
            # print('running threshold={}, processing time={}'.format(threshold, time.time()-start))

        return thresholds, f1_scores, accuracy_scores, precisions, recalls

    def result_with_input_threshold(self, distances, actual_issame, threshold):
        predict_issame = distances < threshold
        f1 = f1_score(actual_issame, predict_issame)
        accuracy = accuracy_score(actual_issame, predict_issame)
        precision = precision_score(actual_issame, predict_issame)
        recall = recall_score(actual_issame, predict_issame)
        return f1, accuracy, precision, recall

    def plot(self,
             thresholds,
             f1_scores,
             accuracy_scores,
             precisions,
             recalls,
             distances,
             actual_issame,
             embs,
             labels):
        save_dir = os.path.join(Config.SAVE_DIR, self.embs_ids_file + '_' + str(time.time()))
        os.mkdir(save_dir)
        opt_idx = np.argmax(f1_scores)
        print_metrics(precisions,
                      recalls,
                      thresholds,
                      opt_idx,
                      os.path.join(save_dir, 'Precision-Recall.jpg'))
        print_metrics(accuracy_scores,
                      f1_scores,
                      thresholds,
                      opt_idx,
                      os.path.join(save_dir, 'Accuracy-F1 Score.jpg'))
        print_distance_histogram(distances,
                                 actual_issame,
                                 thresholds[opt_idx],
                                 os.path.join(save_dir, 'Distances_Histogram.jpg'))
        TSNE_visualize(embs,
                       labels,
                       os.path.join(save_dir, 'TSNE_visualize.jpg'))
        print('plots saved at ', os.path.abspath(save_dir))

    def __get_distance(self, emb_1, emb_2, distance):
        if 'l2' in distance:
            return l2_distance(emb_1, emb_2)
        else:
            return sum_square_distance(emb_1, emb_2)


def main(args):
    # test_dict = {'a':[1,2,3], 'b':[4,5]}
    # result_list = []
    # nrof_folds = args.nrof_folds
    evaluator = Evaluation()
    print('loading data')
    evaluator.load_data(args.data, args.model_2_data)
    if args.import_pairs is not None:
        pairs_pickle = pickle.load(open(args.import_pairs, 'rb'))
        pos_pairs, neg_pairs = pairs_pickle['pos'], pairs_pickle['neg']
    else:
        identity_manager_dict = evaluator.generate_identity_manager(args.limit_nrof_faces)
        # print('generating compare pairs')
        pos_pairs, neg_pairs = evaluator.generate_compare_pairs(identity_manager_dict,
                                                                args.nrof_pairs,
                                                                args.limit_nrof_faces)
        if args.save_pairs:
            save_name = 'pairs-' + str(round(time.time())) + '.pickle'
            save_dict = {'pos': pos_pairs, 'neg': neg_pairs}
            pickle.dump(save_dict, open(save_name, 'wb'))
            print('pos/neg pairs saved in', save_name)
        # print('computing distances')
        distances, actual_issame = evaluator.get_dist_actual_issame_array(pos_pairs, neg_pairs, args.distance)
        if args.threshold:
            # run in test with fixed threshold mode
            f1, accuracy, precision, recall = \
                evaluator.result_with_input_threshold(distances, actual_issame, args.threshold)
            print('f1 = {} at accuracy = {}'.format(f1, accuracy))
            print('precision = {} at recall = {}'.format(precision, recall))

        else:
            # vary threshold to find interesting points
            print('varying thresholds')
            thresholds, f1_scores, accuracy_scores, precisions, recalls = \
                evaluator.calculate_metrics(distances, actual_issame)
            opt_idx = np.argmax(f1_scores)
            print('f1 = {} at accuracy = {}'.format(f1_scores[opt_idx], accuracy_scores[opt_idx]))
            print('precision = {} at recall = {}'.format(precisions[opt_idx], recalls[opt_idx]))
            print('optimal threshold:', thresholds[opt_idx])
            # result_list.append([f1_scores[opt_idx],
            #                     accuracy_scores[opt_idx],
            #                     precisions[opt_idx],
            #                     recalls[opt_idx],
            #                     thresholds[opt_idx]])
            if args.plot:
                # get labels and embs for TSNE visualize
                unique_idxs = pos_pairs + neg_pairs
                unique_idxs = np.unique(np.array(unique_idxs))
                embs = evaluator.features.get_emb(unique_idxs)
                labels = evaluator.features.get_ids(unique_idxs)
                evaluator.plot(thresholds,
                               f1_scores,
                               accuracy_scores,
                               precisions,
                               recalls,
                               distances,
                               actual_issame,
                               embs,
                               labels)
        # print('\n++++++++++++++++++++++++++++++++')
        # print('Average results')
        # results = np.array(result_list)
        # print('f1 = {} at accuracy = {}'.format(results[:,0].sum()/nrof_folds, results[:,1].sum()/nrof_folds))
        # print('precision = {} at recall = {}'.format(results[:,2].sum()/nrof_folds, results[:,3].sum()/nrof_folds))
        # print('optimal threshold:', results[:,4].sum()/nrof_folds)


def parse_arguments(argv):
    parser = argparse.ArgumentParser('parser',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',
                        '-d',
                        help='path to data file contain embeddings and labels')
    parser.add_argument('--model_2_data',
                        '-d2',
                        help='path to data from another model')
    parser.add_argument('--threshold',
                        help='threshold for one time run results',
                        type=float, default=None)
    parser.add_argument('--plot',
                        '-p',
                        help='decide either or not to plot the far, frr graph',
                        action='store_true')
    parser.add_argument('--nrof_pairs',
                        '-np',
                        help='number of positive, negative pairs',
                        type=int, default=float('Inf'))
    parser.add_argument('--distance',
                        '-dt',
                        help='choose the distance to be used: l2 or sum-square',
                        default='l2')
    parser.add_argument('--limit_nrof_faces',
                        '-lnf',
                        help='only take x face for each id to compare',
                        default=float('Inf'), type=int)
    parser.add_argument('--save_pairs',
                        '-sp',
                        help='save generated pos/neg pairs',
                        action='store_true')
    parser.add_argument('--import_pairs',
                        '-ip',
                        help='import generated pos/neg pairs')
    parser.add_argument('--nrof_folds',
                        '-nof',
                        help='number of running folds',
                        type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
