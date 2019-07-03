'''
This is classes, utils for face-tracking system.
We can improve speed some where is '# TODO (@man): this for speed'

Created by @man
Last edit: Jan 9th, 2018
'''

import os
import time
from collections import Counter
import threading
import dlib
import numpy as np
from utils import (wait_recognition_process,
                   PickleUtils,
                   check_overlap)
import cv2
from config import Config
from sklearn import neighbors


class TrackerResult:
    '''
    Doc String
    '''
    def __init__(self):
        self.track_names = []
        self.bounding_boxes = []

    def append_result(self, input_track_name, input_bounding_box):
        '''
        Doc String
        '''
        self.track_names.append(input_track_name)
        self.bounding_boxes.append(input_bounding_box)

    def clear(self):
        '''
        Doc String
        '''
        self.track_names = []
        self.bounding_boxes = []


class TrackerResultsDict:
    '''
    Doc String
    '''
    def __init__(self):
        self.tracker_results_dict = {}

    def update_dict(self, input_fid, input_track_name, input_bounding_box):
        '''
        Doc String
        '''
        if input_fid not in self.tracker_results_dict.keys():
            self.tracker_results_dict[input_fid] = TrackerResult()
        self.tracker_results_dict[input_fid].append_result(input_track_name, input_bounding_box)

    def update_two_dict(self, in_dict):
        '''
        Doc String
        '''
        if in_dict == {}:
            return
        for fid in in_dict.tracker_results_dict:
            if fid not in self.tracker_results_dict.keys():
                self.tracker_results_dict[fid] = TrackerResult()
            in_track_names = in_dict.tracker_results_dict[fid].track_names
            in_bounding_boxes = in_dict.tracker_results_dict[fid].bounding_boxes
            self.tracker_results_dict[fid].track_names += in_track_names
            self.tracker_results_dict[fid].bounding_boxes += in_bounding_boxes


class Tracker:
    '''
    Class docstring here
    '''
    def __init__(self,
                 input_track_id,
                 input_time_stamp,
                 input_bb,
                 input_image,
                 input_embedding,
                 input_angle,
                 input_area,
                 input_frame_id,
                 input_padded_bbox):
        self.track_id = input_track_id
        self.name = None
        self.area = input_area
        self.elements = []
        self.start_time = time.time()
        self.retrack_emb = input_embedding
        self.tag = None
        self.elements.append(TrackerElement(0,
                                            input_time_stamp,
                                            input_bb,
                                            input_image,
                                            input_embedding,
                                            input_angle,
                                            input_frame_id,
                                            input_padded_bbox))

    def update_tracker(self,
                       input_time_stamp,
                       input_bounding_box,
                       input_image,
                       input_embedding,
                       input_angle,
                       input_frame_id,
                       input_padded_bbox):
        '''
        Doc
        '''

        # Use last element
        # self.retrack_emb = input_embedding
        self.elements.append(TrackerElement(len(self.elements),
                                            input_time_stamp,
                                            input_bounding_box,
                                            input_image,
                                            input_embedding,
                                            input_angle,
                                            input_frame_id,
                                            input_padded_bbox))

    def tracker_recognizer(self, matcher, match_mode='voting'):
        '''
        Doc String
        '''
        print('Recognition')
        if len(self.elements) < Config.Track.MIN_NOF_TRACKED_FACES:
            return 'BAD-TRACK'

        predicted_dict = {}
        return_id = ''
        for element in self.elements:
            predict_id, _, min_dist = matcher.match(element.embedding, return_min_dist=True)
            predicted_dict[element.element_id] = (predict_id, min_dist)
            element.set_predict_tup(predict_id, min_dist)
        predicted_ids = [predicted_dict[fid][0] for fid in predicted_dict]
        voted_track_id, voted_num = Counter(predicted_ids).most_common(1)[0]
        voted_rate = voted_num/len(predicted_ids)
        if match_mode == 'combine':
            if voted_rate > Config.Track.COMBINED_MATCHER_RATE:
                match_mode = 'voting'
            else:
                match_mode = 'minmin'

        if match_mode == 'voting':
            return_id = voted_track_id
        if match_mode == 'minmin':
            dicts = {fid: predicted_dict[fid][1]
                     for fid in predicted_dict
                     if predicted_dict[fid][1] != -1}
            return_id = predicted_dict[min(dicts, key=dicts.get)][0]

        return return_id


class TrackerElement:
    '''
    Class docstring here
    '''
    def __init__(self,
                 input_element_id,
                 input_time_stamp,
                 input_bounding_box,
                 input_image,
                 input_embedding,
                 input_angle,
                 input_frame_id,
                 input_padded_bbox):
        self.element_id = input_element_id
        self.time_stamp = input_time_stamp
        self.bounding_box = input_bounding_box
        self.image = input_image
        self.embedding = input_embedding
        self.angle = input_angle
        self.frame_id = input_frame_id
        self.state = 1
        self.predict_tup = (0, 0)
        self.padded_bbox = input_padded_bbox

    def set_state(self, detected_bbs):
        '''
        Set state for this element
        '''
        for detected_bb in detected_bbs:
            if check_overlap(self.bounding_box, detected_bb) > Config.Track.MIN_OVERLAP_IOU:
                return 1
        self.state = 0
        return 0

    def set_predict_tup(self, input_id, input_dist):
        '''
        Set predict tuple
        '''
        self.predict_tup = (input_id, input_dist)


class TrackersHistory:
    '''
    To store good trackers as registered faces
    '''
    def __init__(self):
        self.trackers = {}
        self.current_id = 0
        self.start_time = time.time()
        self.history_matcher = None
        self.labels = []
        self.embs = []

    def match_tracker(self, tracker, match_mode='minrate'):
        '''
        this function matches tracker with history trackers
        '''
        print('match history')
        if match_mode == 'minrate':
            final_rates = []
            for t_element in tracker.elements:
                rates = {}
                for h_fid in self.trackers:
                    dists = []
                    for element in self.trackers[h_fid].elements:
                        dists.append(np.linalg.norm(t_element.embedding - element.embedding))
                    close_dists = [x for x in dists if x < Config.Track.HISTORY_RETRACK_THRESHOLD]
                    rates[h_fid] = len(close_dists)/len(dists)
                high_rate_id = max(rates, key=rates.get)
                if rates[high_rate_id] > Config.Track.HISTORY_RETRACK_MINRATE and \
                        self.trackers[high_rate_id].elements[-1].time_stamp < tracker.start_time:
                    final_rates.append(self.trackers[high_rate_id].name)
            if final_rates == []:
                return Config.Matcher.NEW_FACE
            else:
                return_id, _ = Counter(final_rates).most_common(1)[0]
                return return_id

        if match_mode == 'kdtree':
            predicted_ids = []
            for t_element in tracker.elements:
                dists, inds = self.history_matcher.query([t_element.embedding], k=2)
                dists = np.squeeze(dists)
                inds = np.squeeze(inds)
                if dists[0] < Config.Track.HISTORY_RETRACK_THRESHOLD:
                    predicted_ids.append(self.labels[inds[0]])
                else:
                    predicted_ids.append(Config.Matcher.NEW_FACE)
            return_id, nof_predicted = Counter(predicted_ids).most_common(1)[0]
            if nof_predicted/len(predicted_ids) < Config.Track.HISTORY_RETRACK_MINRATE:
                return Config.Matcher.NEW_FACE
            if return_id != Config.Matcher.NEW_FACE:
                end_times = [self.trackers[fid].elements[-1].time_stamp
                             for fid in self.trackers if self.trackers[fid].name == return_id]
                end_length = len(end_times)
                if end_length > 0:
                    if end_times[-1] > tracker.start_time:
                        return_id = Config.Matcher.NEW_FACE

            return return_id

    def add_tracker(self, tracker, matcher, rabbit_mq):
        '''
        Add tracker to history
        Input: tracker: the tracker u wanna add to history
               matcher: the classifier
        Output: Return tracking results dictionary
        '''

        t_tracker = tracker
        tracker_return_dict = TrackerResultsDict()
        predict_tracker_dict = {}
        if len(tracker.elements) >= Config.Track.MIN_NOF_TRACKED_FACES:

            # Remove some elements under conditions
            # good_elements = [element for element in tracker.elements
            #                  if element.state == 1 and
            #                  abs(element.angle) < 60]
            p_tracker = tracker
            # p_tracker.elements = good_elements

            predicted_id = Config.Matcher.NEW_FACE

            # if (self.trackers != {} and
            #         self.history_matcher is not None):

            #     # match history trackers
            #     print('Match with history')
            #     print('History length: {}'.format(len(self.trackers)))
            #     predicted_id = self.match_tracker(p_tracker, match_mode='kdtree')
            #     if predicted_id != Config.Matcher.NEW_FACE:
            #         # predicted_id += '-retrack' = 1
            #         p_tracker.tag = 'matched-history'

            # if predicted_id == Config.Matcher.NEW_FACE:
            #     # Recognize processed history trackers before adding
            #     predicted_id = p_tracker.tracker_recognizer(matcher,
            #                                                 match_mode='voting')
            #     p_tracker.tag = 'recognized'

            if predicted_id == Config.Matcher.NEW_FACE:

                # t_str = '-{}'.format(time.time())
                predicted_id = 'TCH-' + str(p_tracker.track_id)
                # + '-newface'
                # matcher.plus_one_numofids()
                p_tracker.tag = 'new-face'

            p_tracker.name = predicted_id
            # if Config.Track.MIN_MATCH_DISTACE_OUT:
            #     track_dir = p_tracker.track_id
            #     if not os.path.exists('../data/tracking/match-{}'.format(track_dir)):
            #         os.mkdir('../data/tracking/match-{}'.format(track_dir))
            #     for p_element in p_tracker.elements:
            #         print('Write Tracker {} ID: {}'.format(track_dir,
            #                                                p_element.element_id))
            #         tmp_prt_id = p_element.predict_tup[0]
            #         tmp_prt_dist = p_element.predict_tup[1]
            #         tmp_frame_id = p_element.frame_id
            #         tmp_e_id = p_element.element_id
            #         img_path = '../data/tracking/match-{}/{}_{}_{}_{}.jpg'.format(track_dir,
            #                                                                       tmp_prt_id,
            #                                                                       tmp_prt_dist,
            #                                                                       tmp_frame_id,
            #                                                                       tmp_e_id)
            #         cv2.imwrite(img_path,
            #                     cv2.cvtColor(p_element.image, cv2.COLOR_BGR2RGB))

            # Create t_tracker for extract tracking result
            # with predicted_id more reliable
            t_tracker.name = predicted_id

            recog_list = []
            # Extract history tracking results
            print('Extract tracker: {}'.format(t_tracker.name))
            track_dir = t_tracker.track_id
            if not os.path.exists('../data/tracking/{}'.format(track_dir)):
                os.mkdir('../data/tracking/{}'.format(track_dir))
            for element in t_tracker.elements:
                bbox_str = '_'.join(np.array(element.bounding_box[:], dtype=np.unicode).tolist())
                if Config.Track.FACE_TRACK_IMAGES_OUT:
                    print('Write Tracker {} ID: {}'.format(track_dir,
                                                           element.element_id))
                    img_path = '../data/tracking/{}/{}_{}_{}_{}.jpg'.format(track_dir,
                                                                            t_tracker.name,
                                                                            bbox_str,
                                                                            element.frame_id,
                                                                            element.angle)
                    cv2.imwrite(img_path,
                                cv2.cvtColor(element.image, cv2.COLOR_BGR2RGB))
                if Config.SEND_QUEUE_TO_DASHBOARD and t_tracker.name != 'BAD-TRACK':
                    image_id = '{}_{}_{}_{}'.format(element.frame_id,
                                                    bbox_str,
                                                    t_tracker.area,
                                                    element.time_stamp)
                    recog_list.append((image_id, element.image, t_tracker.name, []))
                    rabbit_mq.send_recognition_result_tch(t_tracker.name,
                                                          element.image,
                                                          rabbit_mq.SEND_QUEUE_TCH_LIVE_RESULT)
                tracker_return_dict.update_dict(element.frame_id,
                                                t_tracker.name,
                                                element.bounding_box)
            print('Saved tracked ids')
            # if Config.SEND_QUEUE_TO_DASHBOARD:
            #     if recog_list:
            #         tmp_recog_list = []
            #         if recog_list[1] != []:
            #             tmp_recog_list.append(recog_list[0])
            #             tmp_recog_list.append(recog_list[1])
            #         rabbit_mq.send_multi_live_reg(tmp_recog_list,
            #                                       rabbit_mq.SEND_QUEUE_LIVE_RESULT)

            # if p_tracker.name != 'BAD-TRACK':
            #     self.trackers[self.current_id] = p_tracker
            #     self.current_id += 1

            #     # rebuild matcher
            #     for element in p_tracker.elements:
            #         self.embs.append(element.embedding)
            #         self.labels.append(p_tracker.name)
            #     self.history_matcher = neighbors.KDTree(np.asarray(self.embs),
            #                                             leaf_size=Config.Matcher.INDEX_LEAF_SIZE,
            #                                             metric='euclidean')
        return tracker_return_dict, predict_tracker_dict

    def extract_register(self):
        '''
        Extract trackers to live dir
        Append image_id to reg_dict_face
        '''
        print('Extract Register Images ...')
        reg_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE)
        if reg_dict is None:
            reg_dict = {}
        extract_fids_list = []
        extract_counter = 0
        for fid in self.trackers:
            if (self.trackers[fid].name == 'BAD-TRACK' or
                    self.trackers[fid].elements == []):
                extract_fids_list.append(fid)
                continue
            if (time.time() - self.trackers[fid].start_time)\
                    < (Config.Track.HISTORY_EXTRACT_TIMER):
                continue
            if self.trackers[fid].tag != 'new-face':
                extract_fids_list.append(fid)
                continue
            limit_counter = 0
            for element in self.trackers[fid].elements:
                if limit_counter >= Config.Track.NOF_REGISTER_IMAGES:
                    break
                # Write images out
                image_id = '{}_{}_{}_{}'.format(self.trackers[fid].name,
                                                element.frame_id,
                                                element.element_id,
                                                element.time_stamp)
                live_tup = ((element.image), (element.embedding))
                PickleUtils.save_pickle('{}/{}.pkl'.format(Config.LIVE_DIR, image_id),
                                        value=live_tup)

                # Save register
                reg_dict[image_id] = self.trackers[fid].name
                limit_counter += 1
            extract_counter += 1
            extract_fids_list.append(fid)
        if extract_fids_list != []:
            # Save register dictionary
            PickleUtils.save_pickle(Config.REG_IMAGE_FACE_DICT_FILE, value=reg_dict)
        return extract_fids_list

    def clear_history(self, extract_fids_list):
        '''
        Clear history as init
        '''
        for fid in extract_fids_list:
            del_instances = [i for i in range(len(self.labels))
                             if self.labels[i] == self.trackers[fid].name]
            if del_instances != []:
                self.labels = [self.labels[i] for i in del_instances]
                self.embs = [self.embs[i] for i in del_instances]
            self.trackers.pop(fid, None)
            self.history_matcher = neighbors.KDTree(np.asarray(self.embs),
                                                    leaf_size=Config.Matcher.INDEX_LEAF_SIZE,
                                                    metric='euclidean')
        self.start_time = time.time()

    def check_time(self, matcher):
        '''
        Check extract time
        '''
        print('History extract elapsed time: {}'.format(time.time() - self.start_time))
        if (time.time() - self.start_time) > Config.Track.HISTORY_CHECK_TIMER:

            # Extract register
            extract_fids_list = self.extract_register()

            # Clear history
            self.clear_history(extract_fids_list)

            # re-build matcher
            build_threading = threading.Thread(target=matcher.update, kwargs={'force': True})
            build_threading.start()

            # Return update matcher flag
            return True

        return False


class TrackersList:  # confuse
    '''
    Class docstring here
    '''
    def __init__(self):
        self.trackers = {}
        self.dlib_trackers = {}
        self.current_track_id = 0
        self.trackers_history = TrackersHistory()

    def get_retrack_tracker(self, emb, frame_id, retrack_mode='lastelement'):
        '''
        Input: self.trackers contains trackers<Trackers>
            emb is a query embedding
        Output: min_dist: distance value
                min_tracker_key: the key of the minimum tracker
        >>>
        xx
        '''
        # between all trackers in self.trackers and emb, return index then
        print('Loop all eleqments in self.trackers to find the one who appears again')
        retrack_trackers = {fid: self.trackers[fid]
                            for fid in self.trackers
                            if self.trackers[fid].elements[-1].frame_id != frame_id}
        if retrack_trackers == {}:
            return -1, -1
        if retrack_mode == 'lastelement':
            dists = []
            for fid in retrack_trackers:
                dists.append(np.linalg.norm(emb - retrack_trackers[fid].retrack_emb))
            min_dist = np.min(dists)
            min_tracker_key = list(retrack_trackers.keys())[np.argmin(dists)]

            if min_dist < Config.Track.RETRACK_THRESHOLD:
                return min_dist, min_tracker_key
            else:
                return -1, -1
        if retrack_mode == 'minrate':
            rates = {}
            for fid in retrack_trackers:
                dists = []
                for element in retrack_trackers[fid].elements:
                    dists.append(np.linalg.norm(emb - element.embedding))
                close_dists = [x for x in dists if x < Config.Track.RETRACK_THRESHOLD]
                rates[fid] = len(close_dists)/len(dists)
            high_rate_id = max(rates, key=rates.get)
            if rates[high_rate_id] > Config.Track.RETRACK_MINRATE:
                return rates[high_rate_id], high_rate_id
            else:
                return -1, -1
        return -1, -1

    def update_trackers_list(self,
                             matched_fid,
                             time_stamp,
                             origin_bb,
                             image,
                             angle,
                             area,
                             input_frame_id,
                             matcher,
                             body_extractor,
                             rabbit_mq):
        '''
        Doc string
        '''
        if matched_fid not in self.trackers.keys():  # edited Jan 10, 2018
            body_emb = body_extractor.extract_feature(image)
            self.trackers[matched_fid] = Tracker(matched_fid,
                                                 time_stamp,
                                                 origin_bb,
                                                 image,
                                                 body_emb,
                                                 angle,
                                                 area,
                                                 input_frame_id,
                                                 None)
            print('Current ID: {}'.format(matched_fid))
            print('Num of trackers: %d' % len(self.trackers))

            # This is the very first step for recognition within x second
            # TODO (@man): this for speed, assign 'detecting'
            # in case we don't need the quick first-step recognition in advance
            if Config.Track.USE_FIRST_STEP_RECOGITION:
                threading_session = threading.Thread(target=wait_recognition_process,
                                                     args=(self.check_recognize_tracker,
                                                           matcher,
                                                           rabbit_mq,
                                                           matched_fid))
                threading_session.start()
            else:
                self.trackers[matched_fid].name = 'DETECTING'

            self.current_track_id += 1

            # Increase the current_track_id counter
            print(matched_fid)
        else:
            # update this tracker
            print('UPDATE TRACK_ID == {}'.format(matched_fid))
            self.trackers[matched_fid].update_tracker(time_stamp,
                                                      origin_bb,
                                                      image,
                                                      None,
                                                      angle,
                                                      input_frame_id,
                                                      None)

    def update_dlib_trackers(self, frame):
        '''
        Input:
        Output:
        Doc string
        '''
        tmpi = frame
        # end_line = 20
        delete_dict = []
        for fid in self.dlib_trackers:

            # crop region of interest to track face in order to optimize runtime
            tracking_quality = self.dlib_trackers[fid][0].update(frame)
            print('Track Quality: %f' % tracking_quality)
            # tracked_position = self.dlib_trackers[fid][0].get_position()
            # x1 = int(tracked_position.left())
            # y1 = int(tracked_position.top())
            # x2 = int(tracked_position.left() + tracked_position.width())
            # y2 = int(tracked_position.top() + tracked_position.height())
            # s_out = '{}: {}'.format(fid, tracking_quality)
            # cv2.rectangle(tmpi,
            #               (x1, y1),
            #               (x2, y2),
            #               (0, 255, 0),
            #               3)
            # cv2.putText(tmpi,
            #             s_out,
            #             (10, end_line),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5,
            #             (255, 255, 255),
            #             2)
            # end_line += 20

            # If the tracking quality is good enough, we must delete
            # this tracker
            if tracking_quality < Config.Track.DLIB_TRACK_QUALITY:
                delete_dict.append(fid)
        for fid in delete_dict:
            print('Removing fid ' + str(fid) + ' from list of trackers')
            self.dlib_trackers.pop(fid, None)
        return tmpi

    def matching_face_with_trackers(self,
                                    frame,
                                    frame_id,
                                    origin_bb,
                                    emb_array,
                                    image,
                                    body_extractor):
        # TODO (manho):
        '''
        Input:
        Output:
        Doc string
        '''
        print('current_track_id: %d' % self.current_track_id)

        x1_origin_bb = int(origin_bb[0])
        y1_origin_bb = int(origin_bb[1])
        x2_origin_bb = int(origin_bb[2])
        y2_origin_bb = int(origin_bb[3])

        # Variable holding information which faceid we
        # matched with
        matched_fid = None

        # Overlap checker
        if self.dlib_trackers != {}:
            ious = {}
            with open('bounding_boxes_checker.txt', 'a') as bb_file:
                bb_file_txt = 'Frame ID: {}\n Current bounding boxex:\n'.format(frame_id)
                bb_file.write(bb_file_txt)
            for fid in self.dlib_trackers:
                if self.dlib_trackers[fid][2] == frame_id:
                    continue
                tracked_position = self.dlib_trackers[fid][0].get_position()
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                track_bb = [t_x - 10, t_y - 20, t_x + t_w + 10, t_y + t_h + 20]
                dlib_bb = [t_x, t_y, t_x + t_w, t_y + t_h]
                with open('bounding_boxes_checker.txt', 'a') as bb_file:
                    bb_file_txt = '{}\n'.format(dlib_bb)
                    bb_file.write(bb_file_txt)
                if check_overlap(track_bb, origin_bb) > 0:
                    real_overlap = check_overlap(dlib_bb, origin_bb)
                    if real_overlap > Config.Track.MIN_OVERLAP_IOU:
                        ious[fid] = check_overlap(dlib_bb, origin_bb)
            with open('bounding_boxes_checker.txt', 'a') as bb_file:
                bb_file_txt = 'Detected bounding box: {}\n'.format(origin_bb)
                bb_file.write(bb_file_txt)
            if ious != {}:
                argmin_fid = max(ious, key=ious.get)
                matched_fid = argmin_fid
            if matched_fid is None:
                with open('bounding_boxes_checker.txt', 'a') as bb_file:
                    bb_file.write('Matched: None\n')
            else:
                with open('bounding_boxes_checker.txt', 'a') as bb_file:
                    bb_file_txt = 'Matched: {}\n'.format(matched_fid)
                    bb_file.write(bb_file_txt)

        # Now loop over all elements in self.trackers and check if
        # there is a matching distance is good enough (< threshold), that
        # means there is a person appears again
        if matched_fid is None:
            list_len = len(self.trackers)
            if list_len > 0:
                body_emb = body_extractor.extract_feature(image)
                # TODO: tracker is dict
                retrack_value, retrack_id = self.get_retrack_tracker(body_emb,
                                                                     frame_id,
                                                                     retrack_mode='lastelement')
                print('Dist or rate: %f' % retrack_value)
                print('Retrack ID: %f' % retrack_id)
                if retrack_value != -1:
                    matched_fid = retrack_id
                    # Create or update the tracker valued retrack_id again
                    # if it doesn't exist in self.dlib_trackers
                    print('Re-creating or update tracker ' + str(matched_fid))

        if matched_fid is None:
            # If no matched fid, then we have to create a new tracker
            print('Creating new tracker ' + str(self.current_track_id))
            matched_fid = self.current_track_id

        # Create and store the tracker
        tracker = dlib.correlation_tracker()
        tracker.start_track(frame,
                            dlib.rectangle(x1_origin_bb,
                                           y1_origin_bb,
                                           x2_origin_bb,
                                           y2_origin_bb))
        self.dlib_trackers[matched_fid] = (tracker, emb_array, frame_id)

        return matched_fid

    def update_trackers_by_tracking(self,
                                    body_extractor,
                                    frame,
                                    area,
                                    input_frame_id,
                                    detected_bbs,
                                    rabbit_mq):
        '''
        Input:
        Output:
        Doc string
        '''
        for fid in self.dlib_trackers:
            tracked_position = self.dlib_trackers[fid][0].get_position()

            # If bb not is inner frame, skip this
            if (tracked_position.left() < 0 or
                    (tracked_position.left() + tracked_position.width()) > frame.shape[1] or
                    tracked_position.top() < 0 or
                    (tracked_position.top() + tracked_position.height()) > frame.shape[0]):
                print('dlib tracker track face is inner of range!')
                continue

            update_bb = np.zeros((4), dtype=np.int32)
            update_bb[0] = tracked_position.left()
            update_bb[1] = tracked_position.top()
            update_bb[2] = tracked_position.left() + tracked_position.width()
            update_bb[3] = tracked_position.top() + tracked_position.height()

            cropped = frame[update_bb[1]:update_bb[3], update_bb[0]:update_bb[2], :]
            emb_array = body_extractor.extract_feature(cropped)

            print('UPDATE TRACK_ID == %d' % fid)
            self.trackers[fid].update_tracker(time.time(),
                                              update_bb,
                                              cropped,
                                              emb_array,
                                              -1,
                                              input_frame_id,
                                              None)
            # self.trackers[fid].elements[-1].set_state(detected_bbs)

    def check_recognize_tracker(self,
                                matcher,
                                rabbit_mq,
                                matched_fid):
        '''
        Input:
        Output:
        Doc string
        '''
        return_flag = True
        print(time.time() - self.trackers[matched_fid].elements[0].time_stamp)
        if ((time.time() - self.trackers[matched_fid].elements[0].time_stamp) >
                Config.Track.RECOG_TIMER and
                self.trackers[matched_fid].name is None):
            return_flag = False
            if len(self.trackers[matched_fid].elements) < Config.Track.MIN_NOF_TRACKED_FACES:
                self.trackers[matched_fid].name = 'BAD-TRACK'
            else:

                # Do recognition here
                predicted_track_id = self.trackers[matched_fid].tracker_recognizer(matcher)
                self.trackers[matched_fid].name = predicted_track_id

            # SEND QUEUE RECOGNITION RESULTS
            if Config.SEND_QUEUE_TO_DASHBOARD:
                tracker_element = self.trackers[matched_fid].elements[-1]
                image_id = '{}_{}_{}_{}'.format(self.trackers[matched_fid].area,
                                                tracker_element.element_id,
                                                tracker_element.time_stamp,
                                                tracker_element.bounding_box)
                recog_list = []
                recog_list.append((image_id, tracker_element.image, predicted_track_id))
                if recog_list:
                    rabbit_mq.send_multi_live_reg(recog_list, rabbit_mq.SEND_QUEUE_LIVE_RESULT)
        return return_flag

    def check_delete_trackers(self, matcher, rabbit_mq, history_mode=True):
        '''
        How long we keep the current trackers?
        Remove tracker if there is no its tracked face for 30s
                          and it's recognized
        '''
        delete_dict = []
        trackers_return_dict = TrackerResultsDict()
        predict_trackers_dict = {}
        for fid in self.trackers:
            elapsed_time = time.time() - self.trackers[fid].elements[-1].time_stamp
            print('Check delete current trackers: {},{}'.format(self.trackers[fid].name,
                                                                elapsed_time))
            if (elapsed_time > Config.Track.CURRENT_EXTRACR_TIMER and
                    self.trackers[fid].name is not None):
                print('Removing fid ' + str(fid) + ' from list of trackers')
                print('len elements: %d' % len(self.trackers[fid].elements))
                delete_dict.append(fid)

        for fid in delete_dict:
            if history_mode:

                # Save tracker to history before remove
                tracker_return_dict, predict_tracker_dict = \
                        self.trackers_history.add_tracker(self.trackers[fid], matcher, rabbit_mq)
                print(tracker_return_dict)
                trackers_return_dict.update_two_dict(tracker_return_dict)
                predict_trackers_dict.update(predict_tracker_dict)
            self.trackers.pop(fid, None)
            self.dlib_trackers.pop(fid, None)
        return trackers_return_dict, predict_trackers_dict
