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
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.detection import Detection as ddet
from matcher import KdTreeMatcher
from config import Config

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='Face Tracking with Deep Sort')
   
    parser.add_argument("--input", dest = 'input', help = 
                        "Video to run detection upon",
                        default = "../../data/1.avi", type = str)
    parser.add_argument("--output", dest = 'output', help = "Output Video", default = "../../data/test1.avi", type = str)
    return parser.parse_args()



args = arg_parse()

vs = cv2.VideoCapture(0)
face_rec_graph = FaceGraph()
coeff_graph = FaceGraph()
extractor_graph = FaceGraph()

detector = MTCNNDetector(face_rec_graph, scale_factor=2)
extractor = FacenetExtractor(extractor_graph)
w = int(vs.get(3))
h = int(vs.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, 15, (w, h))

preprocessor = Preprocessor()
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age = 1000000)

frame_count = 0

while True:
    start = time.time()
    ret, img = vs.read()

    if ret == False:
        break

    if frame_count % 2==0:
        frame_count += 1
        continue

    # detector.detect_face(img)
    
    bounding_boxes, landmarks = detector.detect_face(img)
    if len(bounding_boxes) != 0:
        boxes = [i[:4] for i in bounding_boxes]
        features = []
        bad = []
        for i in range(len(boxes)):
            display_face, str_padded_bbox = CropperUtils.crop_display_face(img, boxes[i])
            cropped_face = CropperUtils.crop_face(img, boxes[i])
            preprocessed_image = preprocessor.process(cropped_face)
            emb_array, _ = extractor.extract_features(preprocessed_image)
            features.append(emb_array)

        boxs = [[i[0],i[1],i[2]-i[0],i[3]-i[1]] for i in boxes]

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(img, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

    cv2.imshow('', img)
    # out.write(img)
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
