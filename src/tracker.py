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
from utils import CropperUtils, check_overlap, wait_recognition_process, PickleUtils
import cv2
from config import Config


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
                 input_face_image,
                 input_embedding,
                 input_angle,
                 input_area,
                 input_frame_id):
        self.track_id = input_track_id
        self.name = None
        self.area = input_area
        self.elements = []
        self.start_time = time.time()
        self.retrack_emb = input_embedding
        self.elements.append(TrackerElement(0,
                                            input_time_stamp,
                                            input_bb,
                                            input_face_image,
                                            input_embedding,
                                            input_angle,
                                            input_frame_id))

    def update_tracker(self,
                       input_time_stamp,
                       input_bounding_box,
                       input_face_image,
                       input_embedding,
                       input_angle,
                       input_frame_id):
        '''
        Doc
        '''

        # Use last element
        self.retrack_emb = input_embedding
        self.elements.append(TrackerElement(len(self.elements),
                                            input_time_stamp,
                                            input_bounding_box,
                                            input_face_image,
                                            input_embedding,
                                            input_angle,
                                            input_frame_id))

    def tracker_recognizer(self, matcher):
        '''
        Doc String
        '''
        print("Recognition")
        if len(self.elements) < Config.Track.MIN_NOF_TRACKED_FACES:
            return 'BAD_TRACK'

        predicted_track_ids = []
        for element in self.elements:
            predict_id, _ = matcher.match(element.embedding)
            predicted_track_ids.append(predict_id)
        voted_track_id, _ = Counter(predicted_track_ids).most_common(1)[0]

        return voted_track_id


class TrackerElement:
    '''
    Class docstring here
    '''
    def __init__(self,
                 input_element_id,
                 input_time_stamp,
                 input_bounding_box,
                 input_face_image,
                 input_embedding,
                 input_angle,
                 input_frame_id):
        self.element_id = input_element_id
        self.time_stamp = input_time_stamp
        self.bounding_box = input_bounding_box
        self.face_image = input_face_image
        self.embedding = input_embedding
        self.angle = input_angle
        self.frame_id = input_frame_id
        self.state = 1

    def set_state(self, detected_bbs):
        '''
        Set state for this element
        '''
        for detected_bb in detected_bbs:
            if check_overlap(self.bounding_box, detected_bb) > Config.Track.MIN_OVERLAP_IOU:
                return 1
        self.state = 0
        return 0


class TrackersHistory:
    '''
    To store good trackers as registered faces
    '''
    def __init__(self):
        self.trackers = {}
        self.current_id = 0
        self.start_time = time.time()

    def add_tracker(self, tracker, matcher, rabbit_mq):
        '''
        Add tracker to history
        Input: tracker: the tracker u wanna add to history
               matcher: the classifier
        Output: Return tracking results dictionary
        '''

        t_tracker = tracker
        tracker_return_dict = TrackerResultsDict()

        if tracker.name != 'BAD_TRACK':

            # Remove some elements under conditions
            good_elements = [element for element in tracker.elements
                             if element.state == 1 and
                             element.angle < 60]
            self.trackers[self.current_id] = tracker
            self.trackers[self.current_id].elements = good_elements

            # Recognize processed history trackers before adding
            predicted_id = self.trackers[self.current_id].tracker_recognizer(matcher)
            self.trackers[self.current_id].name = predicted_id

            # Send RABBIT MQ result to DASHBOARD here
            # TODO (@man): fix dashboard format
            if Config.SEND_QUEUE_TO_DASHBOARD:
                # Face file name if u want to write out
                tracker_element = self.trackers[self.current_id].elements[-1]
                image_id = '{}_{}_{}'.format(self.trackers[self.current_id].area,
                                             self.trackers[self.current_id].track_id,
                                             tracker_element.time_stamp)
                recog_list = []
                recog_list.append((image_id, tracker_element.face_image, predicted_id))
                if recog_list:
                    rabbit_mq.send_multi_live_reg(recog_list, rabbit_mq.SEND_QUEUE_LIVE_RESULT)

            # Create t_tracker for extract tracking result
            # with predicted_id more reliable
            t_tracker.name = predicted_id

        # Extract history tracking results
        if not os.path.exists('../data/tracking/%d' % t_tracker.track_id):
            os.mkdir('../data/tracking/%d' % t_tracker.track_id)
        for element in t_tracker.elements:
            if Config.Track.FACE_TRACK_IMAGES_OUT:
                print("Write Tracker {} ID: {}".format(t_tracker.track_id,
                                                       element.element_id))
                track_dir = t_tracker.track_id
                img_id = "../data/tracking/{}/{}_{}_{}.jpg".format(track_dir,
                                                                   t_tracker.name,
                                                                   element.element_id,
                                                                   element.state)
                cv2.imwrite(img_id,
                            cv2.cvtColor(element.face_image, cv2.COLOR_BGR2RGB))
            print('Saved tracked ids')
            tracker_return_dict.update_dict(element.frame_id,
                                            t_tracker.name,
                                            element.bounding_box)
        self.current_id += 1
        return tracker_return_dict

    def extract_register(self):
        '''
        Extract trackers to live dir
        Append image_id to reg_dict_face
        '''
        print('Extract Register Images ...')
        reg_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE)
        for fid in self.trackers:
            for element in self.trackers[fid].elements:

                # Write images out
                image_id = '{}_{}_{}_{}'.format(self.trackers[fid].name,
                                                element.frame_id,
                                                element.element_id,
                                                element.time_stamp)
                live_tup = ((element.face_image), (element.embedding))
                PickleUtils.save_pickle('{}/{}.pkl'.format(Config.LIVE_DIR, image_id),
                                        value=live_tup)

                # Save register
                reg_dict[image_id] = self.trackers[fid].name

        # Save register dictionary
        PickleUtils.save_pickle(Config.REG_IMAGE_FACE_DICT_FILE, value=reg_dict)

    def clear_history(self):
        '''
        Clear history as init
        '''
        self.trackers = {}
        self.current_id = 0
        self.start_time = time.time()

    def check_time(self):
        '''
        Check extract time
        '''
        print('History extract elapsed time: {}'.format(time.time() - self.start_time))
        if (time.time() - self.start_time) > Config.Track.HISTORY_EXTRACT_TIMER:

            # Extract register
            self.extract_register()

            # Clear history
            self.clear_history()

            # Return update matcher flag
            return True

        return False


class TrackersList:
    '''
    Class docstring here
    '''
    def __init__(self):
        self.trackers = {}
        self.dlib_trackers = {}
        self.current_track_id = 0
        self.trackers_history = TrackersHistory()

    def get_min_tracker(self, emb):
        '''
        Input: self.trackers contains trackers<Trackers>
            emb is a query embedding
        Output: min_dist: distance value
                min_tracker_key: the key of the minimum tracker
        >>>
        xx
        '''
        # between all trackers in self.trackers and emb, return index then
        print("Loop all eleqments in self.trackers to find the one who appears again")
        subs = []
        for fid in self.trackers:
            subs.append(np.subtract(emb, (self.trackers[fid].retrack_emb)))

        dists = np.sum(np.square(subs), 1)
        min_dist = np.min(dists)
        min_tracker_key = list(self.trackers.keys())[np.argmin(dists)]

        return min_dist, min_tracker_key

    def update_trackers_list(self,
                             matched_fid,
                             origin_bb,
                             face_image,
                             emb_array,
                             angle,
                             area,
                             input_frame_id,
                             matcher,
                             rabbit_mq):
        '''
        Doc string
        '''
        if matched_fid not in self.trackers.keys():  # edited Jan 10, 2018
            self.trackers[matched_fid] = Tracker(matched_fid,
                                                 time.time(),
                                                 origin_bb,
                                                 face_image,
                                                 emb_array,
                                                 angle,
                                                 area,
                                                 input_frame_id)
            print("Current ID: %d" % matched_fid)
            print("track_id: %d" % self.trackers[matched_fid].track_id)
            print("Num of trackers: %d" % len(self.trackers))

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
            print("Num of Current trackers: %d" % len(self.dlib_trackers))
            print("Current ID: %d" % self.current_track_id)
            print(matched_fid)
        else:
            # update this tracker
            print('UPDATE TRACK_ID == %d' % matched_fid)
            self.trackers[matched_fid].update_tracker(time.time(),
                                                      origin_bb,
                                                      face_image,
                                                      emb_array,
                                                      angle,
                                                      input_frame_id)

    def update_dlib_trackers(self, frame):
        '''
        Input:
        Output:
        Doc string
        '''
        delete_dict = []
        for fid in self.dlib_trackers:

            # crop region of interest to track face in order to optimize runtime
            tracking_quality = self.dlib_trackers[fid].update(frame)
            print("Track Quality: %f" % tracking_quality)

            # If the tracking quality is good enough, we must delete
            # this tracker
            if tracking_quality < 7:
                delete_dict.append(fid)
        for fid in delete_dict:
            print("Removing fid " + str(fid) + " from list of trackers")
            self.dlib_trackers.pop(fid, None)

    def matching_face_with_trackers(self,
                                    frame,
                                    origin_bb,
                                    emb_array):
        # TODO (manho):
        '''
        Input:
        Output:
        Doc string
        '''
        print('current_track_id: %d' % self.current_track_id)

        x_origin_bb = int(origin_bb[0])
        y_origin_bb = int(origin_bb[1])
        w_origin_bb = int(origin_bb[2]) - x_origin_bb
        h_origin_bb = int(origin_bb[3]) - y_origin_bb

        # Variable holding information which faceid we
        # matched with
        matched_fid = None

        for fid in self.dlib_trackers:
            tracked_position = self.dlib_trackers[fid].get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            # Check overlap
            if check_overlap([t_x, t_y, t_x + t_w, t_y + t_h],
                             origin_bb) > Config.Track.MIN_OVERLAP_IOU:
                matched_fid = fid

        # Now loop over all elements in self.trackers and check if
        # there is a matching distance is good enough (< threshold), that
        # means there is a person appears again
        if matched_fid is None:
            list_len = len(self.trackers)
            if list_len > 0:
                # TODO: tracker is dict
                min_dist, min_track_id = self.get_min_tracker(emb_array)
                print("Min-dist: %f" % min_dist)
                print("Min-dist-id: %f" % min_track_id)
                if min_dist < 0.75:
                    matched_fid = min_track_id
                    # Create or update the tracker valued min_track_id again
                    # if it doesn't exist in self.dlib_trackers
                    print("Re-creating or update tracker " + str(matched_fid))

        if matched_fid is None:
            # If no matched fid, then we have to create a new tracker
            print("Creating new tracker " + str(self.current_track_id))
            matched_fid = self.current_track_id

        # Create and store the tracker
        tracker = dlib.correlation_tracker()
        tracker.start_track(frame,
                            dlib.rectangle(x_origin_bb,
                                           y_origin_bb,
                                           x_origin_bb + w_origin_bb,
                                           y_origin_bb + h_origin_bb))
        self.dlib_trackers[matched_fid] = tracker

        return matched_fid

    def update_trackers_by_tracking(self,
                                    face_extractor,
                                    preprocessor,
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
            tracked_position = self.dlib_trackers[fid].get_position()

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

            cropped = CropperUtils.crop_face(frame, update_bb)
            display_face, padded_bb = CropperUtils.crop_display_face(frame, update_bb)

            preprocessed_image = preprocessor.process(cropped)
            emb_array = face_extractor.extract_features(preprocessed_image)

            print('UPDATE TRACK_ID == %d' % fid)
            self.trackers[fid].update_tracker(time.time(),
                                              update_bb,
                                              cropped,
                                              emb_array,
                                              -1,
                                              input_frame_id)
            self.trackers[fid].elements[-1].set_state(detected_bbs)

            if Config.Track.TRACKING_QUEUE_CAM_TO_CENTRAL:
                track_tuple = (fid, display_face, emb_array, area, time.time(), padded_bb)
                rabbit_mq.send_tracking(track_tuple, rabbit_mq.RECEIVE_CAM_WORKER_TRACKING_QUEUE)

    def check_recognize_trackers(self, recogntion_function, rabbit_mq):
        '''
        Input:
        Output:
        Doc string
        '''
        return_flag = True

        for fid in self.trackers:
            print(time.time() - self.trackers[fid].elements[0].time_stamp)
            if ((time.time() - self.trackers[fid].elements[0].time_stamp) >
                    Config.Track.RECOG_TIMER and
                    self.trackers[fid].name is None):
                return_flag = False

                # Do recognition here
                predicted_track_id = self.trackers[fid].tracker_recognizer(recogntion_function)
                self.trackers[fid].name = predicted_track_id

                # SEND QUEUE RECOGNITION RESULTS
                if Config.SEND_QUEUE_TO_DASHBOARD:
                    tracker_element = self.trackers[fid].elements[-1]
                    image_id = '{}_{}_{}_{}'.format(self.trackers[fid].area,
                                                    tracker_element.element_id,
                                                    tracker_element.time_stamp,
                                                    tracker_element.bounding_box)
                    recog_list = []
                    recog_list.append((image_id, tracker_element.face_image, predicted_track_id))
                    if recog_list:
                        rabbit_mq.send_multi_live_reg(recog_list, rabbit_mq.SEND_QUEUE_LIVE_RESULT)
        return return_flag

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
                self.trackers[matched_fid].name = 'BAD_TRACK'
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
                recog_list.append((image_id, tracker_element.face_image, predicted_track_id))
                if recog_list:
                    rabbit_mq.send_multi_live_reg(recog_list, rabbit_mq.SEND_QUEUE_LIVE_RESULT)
        return return_flag

    def check_delete_trackers(self, matcher, rabbit_mq):
        '''
        How long we keep the current trackers?
        Remove tracker if there is no its tracked face for 30s
                          and it's recognized
        '''
        delete_dict = []
        trackers_return_dict = TrackerResultsDict()
        for fid in self.trackers:
            elapsed_time = time.time() - self.trackers[fid].elements[-1].time_stamp
            print('Check delete: {},{}'.format(self.trackers[fid].name,
                                               elapsed_time))
            if ((time.time() - self.trackers[fid].elements[-1].time_stamp) > 30 and
                    self.trackers[fid].name is not None):
                print("Removing fid " + str(fid) + " from list of trackers")
                print("len elements: %d" % len(self.trackers[fid].elements))
                delete_dict.append(fid)

        for fid in delete_dict:

            # Save tracker to history before remove
            tracker_return_dict = self.trackers_history.add_tracker(self.trackers[fid],
                                                                    matcher,
                                                                    rabbit_mq)
            print(tracker_return_dict)
            trackers_return_dict.update_two_dict(tracker_return_dict)
            self.trackers.pop(fid, None)
            self.dlib_trackers.pop(fid, None)
        return trackers_return_dict
