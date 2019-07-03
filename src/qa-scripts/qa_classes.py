import os
import pickle
from collections import OrderedDict
from operator import itemgetter


class Features:
    def __init__(self, embs_ids_file):
        if not os.path.exists(embs_ids_file):
            raise Exception('Can''t locate {}'.format(embs_ids_file))
        with open(embs_ids_file, 'rb') as f:
            data = pickle.load(f)
        self.embs = data['embs']
        self.ids = data['labels']
        self.image_ids = data['image_ids']

    def get_emb(self, idx):
        return self.embs[idx].squeeze()

    def get_id(self, idx):
        return self.ids[idx]

    def get_ids(self, idxs):
        labels = []
        for idx in idxs:
            labels.append(self.ids[idx])
        return labels

    def get_tracker_id(self, idx):
        return self.image_ids[idx].split('_')[0]

    def get_nrof_embs(self):
        return len(self.ids)

    def get_frame_id(self, idx):
        return float(self.image_ids[idx].split('_')[-5])


class IdentityManager:
    def __init__(self):
        self.tracker_dict = {}

    def add_idx(self, tracker_id, idx):
        if tracker_id not in self.tracker_dict:
            self.tracker_dict[tracker_id] = []
        self.tracker_dict[tracker_id].append(idx)

    def get_1st_tracker(self):
        return self.tracker_dict[self.first_tracker]

    def update_curret_1st_tracker_id(self):
        tracker_id_list = [int(tracker_id) for tracker_id in self.tracker_dict.keys()]
        self.first_tracker = str(min(tracker_id_list))

    def sort_idxs_by_frame_id(self, features):
        for tracker_id, idxs_list in self.tracker_dict.items():
            idx_frame_id_dict = {idx: features.get_frame_id(idx) for idx in idxs_list}
            # print(sorted(idx_frame_id_dict.items(), key=itemgetter(1)))
            new_idxs_sorted_dict = OrderedDict(sorted(idx_frame_id_dict.items(), key=itemgetter(1)))
            self.tracker_dict[tracker_id] = list(new_idxs_sorted_dict.keys())

    def get_nrof_idxs(self):
        nrof_idxs = 0
        for _, idxs_list in self.tracker_dict.items():
            nrof_idxs += len(idxs_list)
        return nrof_idxs

    def get_other_trackers_list(self, nrof_sample=-2):
        other_tracker_list = []
        for tracker_id, idxs_list in self.tracker_dict.items():
            if tracker_id != self.first_tracker:
                min_nrof_sample = min(nrof_sample, len(idxs_list))
                other_tracker_list.append(idxs_list[:min_nrof_sample])
        return other_tracker_list

    
