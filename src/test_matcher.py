from __future__ import division, print_function, absolute_import

import os
import time
from timeit import time
import warnings
import sys
import cv2
import numpy as np
import argparse
import glob
from PIL import Image

from scipy import misc
from utils import (FaceAngleUtils,
                   CropperUtils,
                   is_inner_bb,
                   clear_session_folder,
                   create_if_not_exist,
                   PickleUtils)
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from preprocess import Preprocessor, align_and_crop
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from matcher import KdTreeMatcher
from config import Config
from collections import Counter

distance_threshold = 0.68
major_rate = 0.5
valid_rate = 0.5

def predict_tracker_id(major_rate, n_valid_element_rate, distance_threshold, predicted_ids, predicted_dists):
    n_valid_elements = 0
    nof_predicted = 0
    if(len(predicted_ids) > 0):
        final_predict_id , nof_predicted = Counter(predicted_ids).most_common(1)[0]
        for i, id in enumerate(predicted_ids):
            if id == final_predict_id and predicted_dists[i] <= distance_threshold:
                n_valid_elements +=1

        #verify by major_rate and valid element rate
        if nof_predicted / len(predicted_ids) < major_rate \
                or n_valid_elements / nof_predicted < n_valid_element_rate:
            final_predict_id = Config.Matcher.NEW_FACE
    else:
        final_predict_id = Config.Matcher.NEW_FACE

    return final_predict_id    

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='Face Tracking with Deep Sort')
   
    parser.add_argument("--input", dest = 'input', help = 
                        "Video to run detection upon",
                        default = "../../data/1.avi", type = str)
    parser.add_argument("--output", dest = 'output', help = "Output Video", default = "../../data/out_1.avi", type = str)
    return parser.parse_args()

def handle_filters(point, coeff_extractor, preprocessed_image):
    is_good_face = True
    # Calculate angle
    yaw_angle = FaceAngleUtils.calc_angle(point)
    pitch_angle = FaceAngleUtils.calc_face_pitch(point)
    if abs(yaw_angle) > Config.Filters.YAW:
        # img_path = '../data/outofangle/yaw_{}_{}_{}.jpg'.format(face_info.frame_id,
        #                                                         bbox_str,
        #                                                         abs(yaw_angle))
        # cv2.imwrite(img_path,
        #             cv2.cvtColor(face_info.display_image, cv2.COLOR_BGR2RGB))
        is_good_face = False
    if abs(pitch_angle) > Config.Filters.PITCH:
        # img_path = '../data/outofangle/pitch_{}_{}_{}.jpg'.format(face_info.frame_id,
        #                                                           bbox_str,
        #                                                           abs(pitch_angle))
        # cv2.imwrite(img_path,
        #             cv2.cvtColor(face_info.display_image, cv2.COLOR_BGR2RGB))
        is_good_face = False

    _, coeff_score = coeff_extractor.extract_features(preprocessed_image)
    if coeff_score < Config.Filters.COEFF:
        # img_path = '../data/notenoughcoeff/{}_{}_{}.jpg'.format(face_info.frame_id,
        #                                                         bbox_str,
        #                                                         coeff_score)
        # cv2.imwrite(img_path,
        #             cv2.cvtColor(face_info.display_image, cv2.COLOR_BGR2RGB))
        is_good_face = False
    # else:
    #     with open('../data/coeff_log.txt', 'a') as f:
    #         f.write('{}_{}_{}, coeff: {}\n'.format(bbox_str,
    #                                                face_info.frame_id,
    #                                                face_info.str_padded_bbox,
    #                                                coeff))
    return is_good_face

args = arg_parse()
vs = cv2.VideoCapture(args.input)

face_rec_graph = FaceGraph()
coeff_graph = FaceGraph()
extractor_graph = FaceGraph()

detector = MTCNNDetector(face_rec_graph, scale_factor=2)
extractor = FacenetExtractor(extractor_graph)
w = int(vs.get(3))
h = int(vs.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, 15, (w, h))

coeff_extractor = FacenetExtractor(coeff_graph, model_path=Config.COEFF_DIR)
preprocessor = Preprocessor()
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
matcher = KdTreeMatcher()

frame_count = 0

registered = {}

while True:
    ret, img = vs.read()

    if ret == False:
        break

    start = time.time()
    # if frame_count % 2==0:
    #     # tracker.predict()
    #     # for track in tracker.tracks:
    #     #     if track.is_confirmed() and track.time_since_update > 1:
    #     #         continue

    #     #     bbox = track.to_tlbr()
    #     #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
    #     #     cv2.putText(img, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

    #     # out.write(img)
    #     frame_count += 1
    #     continue
        
    # Start dectecting faces
    bounding_boxes, landmarks = detector.detect_face(img)

    if len(bounding_boxes) != 0:
        boxes = [i[:4] for i in bounding_boxes]
        features = []
        bad = []

        # Getting features of detected faces
        for i in range(len(boxes)):
            display_face, str_padded_bbox = CropperUtils.crop_display_face(img, boxes[i])
            cropped_face = CropperUtils.crop_face(img, boxes[i])
            preprocessed_image = preprocessor.process(cropped_face)
            emb_array, _ = extractor.extract_features(preprocessed_image)
            features.append(emb_array)
        
        # for i in range(len(features)-1,-1,-1):
        #     for j in registered:
        #         dist = np.sqrt(np.sum(np.square(np.subtract(features[i],np.mean(registered[j],axis=0))))) 
        #         if dist < 0.68:
        #             cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),(255,255,255), 2)
        #             cv2.putText(img, str(j),(int(boxes[i][0]), int(boxes[i][1])),0, 5e-3 * 200, (0,255,0),2)
        #             features.pop(i)
        #             boxes.pop(i)
        #             break
        
        # Adding faces and its features to detections
        boxs = [[i[0],i[1],i[2]-i[0],i[3]-i[1]] for i in boxes]
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        deleted = tracker.update(detections)

        # Save features of lost faces
        for t in deleted:
            if len(t.features) > 20:
                matcher.update(t.features, [t.track_id]*len(t.features))
                print(len(t.features))
                # registered[t.track_id] = registered.get(t.track_id,[]) + t.features
                # registered[t.track_id] = t.features

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1:
                continue
            if len(track.features) == 10:
                predicted_ids = []
                predicted_dists = []
                for feature in track.features:
                    top_ids, dists = matcher.match([feature], threshold = distance_threshold, top_matches = 10,\
                                                    return_dists=True, always_return_closest = True)
                    predicted_ids += top_ids
                    predicted_dists += dists
                #predicted_ids = [regdict[id]["face_id"] for id in predicted_ids]
                final_predict_id = predict_tracker_id(major_rate, valid_rate, distance_threshold,\
                                                                        predicted_ids, predicted_dists)
                print(predicted_ids)
                print(predicted_dists)
                print(final_predict_id)
                
                if final_predict_id != "NEW_FACE":
                    track.track_id = final_predict_id
                    track.state = 1

                # if track.track_id > len(registered):
                #     for j in registered:
                #         dist = np.sqrt(np.sum(np.square(np.subtract(np.mean(track.features,axis=0),np.mean(registered[j],axis=0)))))
                #         if dist < 0.6:
                #             track.track_id = j
                #             track.state = 1
                #             break

            bbox = track.to_tlbr()
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(img, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

    # cv2.imshow('', img)
    out.write(img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_count +=1

    end = time.time()
    sec = end-start
    # print(sec)
    # print(frame_count)
    print('FPS: {0}'.format((1.0/sec)))

vs.release()

cv2.destroyAllWindows()
