
import os
import sys
import time
import random
import pickle
import argparse
import itertools
import numpy as np
from qa_config import Config
from qa_utils import l2_distance, sum_square_distance, Features, chop_extension
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

    def generate_id_emb_index_dict(self, limit_nrof_faces=None):
        id_face_dict = {}
        # group all embs to its id
        for i in range(self.features.get_nrof_embs()):
            _id = self.features.get_id(i)
            if _id in id_face_dict:
                id_face_dict[_id].append(i)
            else:
                id_face_dict[_id] = [i]

        if limit_nrof_faces is not None:
            for key, value in id_face_dict.items():
                current_len = len(value)
                nrof_faces_to_compare = min(current_len, limit_nrof_faces)
                id_face_dict[key] = random.sample(value, nrof_faces_to_compare)
        # sort dictionary buy nrof faces of each id
        return_dict = OrderedDict()
        sorted_items = sorted(id_face_dict.items(), key=lambda x: len(x[1]), reverse=True)
        for k, v in sorted_items:
            return_dict[k] = v
        return return_dict

    def generate_compare_pairs(self, id_face_dict, nrof_pairs, limit_nrof_faces):
        pos_pairs = []
        neg_pairs = []
        ids = list(id_face_dict.keys())
        nrof_id = len(ids)
        for idx in range(nrof_id):
            this_id = ids[idx]
            this_id_faces = id_face_dict[this_id]
            # get positive pairs from faces of this id
            this_pos_pairs = list(itertools.combinations(this_id_faces, r=2))
            pos_pairs += this_pos_pairs
            # get negative pairs by product all this id faces to other ids faces
            other_ids_faces = []
            for k in ids[idx + 1:]:
                other_ids_faces += id_face_dict[k]
            # only generate nrof neg pairs equal to nrof pos pairs
            if other_ids_faces:
                this_neg_pairs = []
                for i in range(len(this_pos_pairs)):
                    this_neg_pairs.append((random.choice(other_ids_faces),
                                           random.choice(this_id_faces)))
                neg_pairs += this_neg_pairs
        min_nrof_pairs = min(len(pos_pairs), len(neg_pairs))
        nrof_pairs = min(nrof_pairs, min_nrof_pairs)
        return_pos_pairs = random.sample(pos_pairs, nrof_pairs)
        return_neg_pairs = random.sample(neg_pairs, nrof_pairs)

        print('number of pair: ', nrof_pairs)
        return return_pos_pairs, return_neg_pairs

    def get_dist_actual_issame_array(self, pos_pairs, neg_pairs, distance_type='l2'):
        distances = []
        actual_issame = []
        # if input two embs file from two folders
        if hasattr(self, 'model_2_features'):
            for (emb_idx_1, emb_idx_2) in pos_pairs:
                emb_1a = self.features.get_emb(emb_idx_1)
                emb_2a = self.features.get_emb(emb_idx_2)
                emb_1b = self.model_2_features.get_emb(emb_idx_1)
                emb_2b = self.model_2_features.get_emb(emb_idx_2)
                emb_1 = np.vstack([emb_1a, emb_1b])
                emb_2 = np.vstack([emb_2a, emb_2b])
                dist = self.__get_distance(emb_1, emb_2, distance_type)
                distances.append(dist)
                actual_issame.append(True)

            for (emb_idx_1, emb_idx_2) in neg_pairs:
                emb_1a = self.features.get_emb(emb_idx_1)
                emb_2a = self.features.get_emb(emb_idx_2)
                emb_1b = self.model_2_features.get_emb(emb_idx_1)
                emb_2b = self.model_2_features.get_emb(emb_idx_2)
                emb_1 = np.vstack([emb_1a, emb_1b])
                emb_2 = np.vstack([emb_2a, emb_2b])
                dist = self.__get_distance(emb_1, emb_2, distance_type)
                distances.append(dist)
                actual_issame.append(False)

            return np.array(distances), np.array(actual_issame)

        # else do normal, only take 1 embs data
        for (emb_idx_1, emb_idx_2) in pos_pairs:
            emb_1 = self.features.get_emb(emb_idx_1)
            emb_2 = self.features.get_emb(emb_idx_2)
            dist = self.__get_distance(emb_1, emb_2, distance_type)
            distances.append(dist)
            actual_issame.append(True)

        for (emb_idx_1, emb_idx_2) in neg_pairs:
            emb_1 = self.features.get_emb(emb_idx_1)
            emb_2 = self.features.get_emb(emb_idx_2)
            dist = self.__get_distance(emb_1, emb_2, distance_type)
            distances.append(dist)
            actual_issame.append(False)

        return np.array(distances), np.array(actual_issame)

    # def find_all_tp_fp(self, distances, actual_issame):
    #     np_distances = np.array((distances))
    #     sort_index = np_distances.argsort()
    #     sorted_distances = [distances[i] for i in sort_index]
    #     sorted_actual_issame = np.array(([actual_issame[i] for i in sort_index]))
    #     nrof_pos_pairs = sorted_actual_issame.sum()
    #     nrof_neg_pairs = sorted_actual_issame.size - nrof_pos_pairs

    #     result_array = []
    #     for idx, threshold in enumerate(sorted_distances):
    #         tp, fp = self.__count_tp_fp(sorted_actual_issame[:idx])
    #         tn = nrof_neg_pairs - fp
    #         fn = nrof_pos_pairs - tp
    #         result_array.append(Result(threshold, tp, fp, tn, fn))
    #     return result_array

    def calculate_metrics(self, distances, actual_issame):
        f1_scores = []
        accuracy_scores = []
        precisions = []
        recalls = []

        min_distance = round(distances.min(), 2)
        max_distance = round(distances.max(), 2)
        thresholds = np.arange(min_distance + 0.01, max_distance - 0.01, 0.01)
        for threshold in thresholds:
            predict_issame = distances < threshold
            f1_scores.append(f1_score(actual_issame, predict_issame))
            accuracy_scores.append(accuracy_score(actual_issame, predict_issame))
            precisions.append(precision_score(actual_issame, predict_issame))
            recalls.append(recall_score(actual_issame, predict_issame))

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
             actual_issame):
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
        TSNE_visualize(self.features.embs.squeeze(),
                       self.features.ids,
                       os.path.join(save_dir, 'TSNE_visualize.jpg'))
        print('plots saved at ', os.path.abspath(save_dir))

    def __get_distance(self, emb_1, emb_2, distance):
        if 'l2' in distance:
            return l2_distance(emb_1, emb_2)
        else:
            return sum_square_distance(emb_1, emb_2)


def divide_id_face_dict(id_face_dict, divide_factor):
    new_dict = OrderedDict()
    for k,v in id_face_dict.items():
        nrof_pop_elements = int(len(v)/divide_factor)
        new_dict[k] = []
        for i in range(nrof_pop_elements):
            idx = random.randint(0, len(v) - 1)
            new_dict[k].append(v.pop(idx))

    return new_dict, id_face_dict

def main(args):
    # test_dict = {'a':[1,2,3], 'b':[4,5]}
    result_list = []
    nrof_folds = args.nrof_folds
    evaluator = Evaluation()
    print('loading data')
    evaluator.load_data(args.data, args.model_2_data)
    if args.import_pairs is not None:
        pairs_pickle = pickle.load(open(args.import_pairs, 'rb'))
        pos_pairs, neg_pairs = pairs_pickle['pos'], pairs_pickle['neg']
    else:
        all_id_face_dict = evaluator.generate_id_emb_index_dict(args.limit_nrof_faces)
        # print('generating compare pairs')
        for fold_number in range(nrof_folds):
            print('Running Fold', fold_number)
            fold_dict, all_id_face_dict = divide_id_face_dict(all_id_face_dict,
                                                              nrof_folds - fold_number)
            pos_pairs, neg_pairs = evaluator.generate_compare_pairs(fold_dict,
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
                result_list.append([f1_scores[opt_idx],
                                    accuracy_scores[opt_idx],
                                    precisions[opt_idx],
                                    recalls[opt_idx],
                                    thresholds[opt_idx]])
                if args.plot:
                    evaluator.plot(thresholds,
                                   f1_scores,
                                   accuracy_scores,
                                   precisions,
                                   recalls,
                                   distances,
                                   actual_issame)
        print('\n++++++++++++++++++++++++++++++++')
        print('Average results')
        results = np.array(result_list)
        print('f1 = {} at accuracy = {}'.format(results[:,0].sum()/nrof_folds, results[:,1].sum()/nrof_folds))
        print('precision = {} at recall = {}'.format(results[:,2].sum()/nrof_folds, results[:,3].sum()/nrof_folds))
        print('optimal threshold:', results[:,4].sum()/nrof_folds)


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
