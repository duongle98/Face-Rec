'''
Perform detection + tracking + recognition
Run: python3 generic_detection_tracking.py -c <camera_path> default <rabbit_mq>
                                                                    (for reading frames)
                                           -a <area> default 'None'
                                                                    (for area)
                                           -wi True default False
                                                                    (for writing all face-tracks)
                                           -vo True default False
                                                                    (for write tracking video)
'''
import argparse
import time
from frame_reader import URLFrameReader, RabbitFrameReader
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from matcher import KdTreeMatcher
from tf_graph import FaceGraph
from config import Config
from rabbitmq import RabbitMQ
from video_writer import VideoHandle
from tracker import TrackersList, TrackerResultsDict
from preprocess import Preprocessor
from utils import FaceAngleUtils, CropperUtils

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))


def generic_function(cam_url, area):
    '''
    This is main function
    '''
    print("Generic function")
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))
    # Variables for tracking faces
    frame_counter = 0

    # Variables holding the correlation trackers and the name per faceid
    list_of_trackers = TrackersList()

    face_rec_graph = FaceGraph()
    face_extractor = FacenetExtractor(face_rec_graph)
    detector = MTCNNDetector(face_rec_graph)
    preprocessor = Preprocessor()
    matcher = KdTreeMatcher()
    track_results = TrackerResultsDict()
    if Config.CALC_FPS:
        start_time = time.time()
    if args.cam_url is not None:
        frame_reader = URLFrameReader(args.cam_url, scale_factor=1.5)
    else:
        frame_reader = RabbitFrameReader(rabbit_mq)
    video_out = None
    if Config.Track.TRACKING_VIDEO_OUT:
        video_out_fps, video_out_w, video_out_h, _ = frame_reader.get_info()
        video_out = VideoHandle('../data/tracking_video_out.avi',
                                video_out_fps,
                                int(video_out_w),
                                int(video_out_h))

    try:
        while True:  # frame_reader.has_next():
            frame = frame_reader.next_frame()
            if frame is None:
                print("Waiting for the new image")
                trackers_return_dict = list_of_trackers.check_delete_trackers(matcher, rabbit_mq)
                track_results.update_two_dict(trackers_return_dict)
                continue

            print("Frame ID: %d" % frame_counter)

            if Config.Track.TRACKING_VIDEO_OUT:
                video_out.tmp_video_out(frame)
            if Config.CALC_FPS:
                fps_counter = time.time()

            list_of_trackers.update_dlib_trackers(frame)

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                origin_bbs, points = detector.detect_face(frame)
                for i, origin_bb in enumerate(origin_bbs):
                    display_face, _ = CropperUtils.crop_display_face(frame, origin_bb)
                    cropped_face = CropperUtils.crop_face(frame, origin_bb)

                    # Calculate embedding
                    preprocessed_image = preprocessor.process(cropped_face)
                    emb_array = face_extractor.extract_features(preprocessed_image)

                    # Calculate angle
                    angle = FaceAngleUtils.calc_angle(points[:, i])

                    # TODO: refractor matching_detected_face_with_trackers
                    matched_fid = list_of_trackers.matching_face_with_trackers(frame,
                                                                               origin_bb,
                                                                               emb_array)

                    # Update list_of_trackers
                    list_of_trackers.update_trackers_list(matched_fid,
                                                          origin_bb,
                                                          display_face,
                                                          emb_array,
                                                          angle,
                                                          area,
                                                          frame_counter,
                                                          matcher,
                                                          rabbit_mq)

            trackers_return_dict = list_of_trackers.check_delete_trackers(matcher, rabbit_mq)
            track_results.update_two_dict(trackers_return_dict)

            # Check extract trackers history time
            list_of_trackers.trackers_history.check_time()

            frame_counter += 1
            if Config.CALC_FPS:
                print("FPS: %f" % (1/(time.time() - fps_counter)))
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
    except KeyboardInterrupt:
        print('Keyboard Interrupt !!! Release All !!!')
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format((frame_counter + 1)/(time.time() - start_time)))
        frame_reader.release()
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
            video_out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('For demo only',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',
                        '--cam_url',
                        help='your camera ip address',
                        default=None)
    parser.add_argument('-a',
                        '--area',
                        help='The area that your ip camera is recording at',
                        default='None')
    parser.add_argument('-wi',
                        '--write_images',
                        help='Write all face-tracks out following the path data/tracking',
                        default=False)
    parser.add_argument('-vo',
                        '--video_out',
                        help='Write tracking video out following the path data/tracking',
                        default=False)
    args = parser.parse_args()

    # Run
    if args.write_images == 'True':
        Config.Track.FACE_TRACK_IMAGES_OUT = True
    if args.video_out == 'True':
        Config.Track.TRACKING_VIDEO_OUT = True

    generic_function(args.cam_url, args.area)
