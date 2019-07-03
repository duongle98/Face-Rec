from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from utils import show_frame, CropperUtils
from preprocess import Preprocessor
from matcher import KdTreeMatcher
from frame_reader import URLFrameReader
import time

matcher = KdTreeMatcher()
face_graph = FaceGraph()
face_detector = MTCNNDetector(face_graph)
feature_extractor = FacenetExtractor(face_graph)
preprocessor = Preprocessor()
frame_reader = URLFrameReader(cam_url=0, scale_factor=2)

while frame_reader.has_next():
    frame = frame_reader.next_frame()
    bouncing_boxes, landmarks = face_detector.detect_face(frame)
    nrof_faces = len(bouncing_boxes)
    start = time.time()
    for i in range(nrof_faces):
        cropped = CropperUtils.crop_face(frame, bouncing_boxes[i])
        display_face, padded_bb_str = CropperUtils.crop_display_face(frame, bouncing_boxes[i])
        reverse_face = CropperUtils.reverse_display_face(display_face, padded_bb_str)
        process_img = preprocessor.process(cropped)
        show_frame(reverse_face, 'Reverse')
        show_frame(cropped, 'Cropped')
        emb = feature_extractor.extract_features(process_img)
        predict_id, top_match_ids = matcher.match(emb)
        print('Predict', predict_id)

    show_frame(frame, 'nn')
    end = time.time()
    print('Fps', 1 / (end - start))
    start = end
