import argparse
import time
import cv2
from frame_reader import URLFrameReader, RabbitFrameReader
from config import Config
from rabbitmq import RabbitMQ
from video_writer import VideoHandle
from human_tracker import TrackersList, TrackerResultsDict
from trinet_embed import BodyExtractor
from darkflow.net.build import TFNet
from utils import (calc_bb_size,
                   PickleUtils,
                   clear_tracking_folder,
                   is_inner_bb)

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))


def draw_results(image, detect_results):
    '''
    yeh
    '''
    tmpi = image
    for result in detect_results:
        x1 = result['topleft']['x']
        y1 = result['topleft']['y']
        x2 = result['bottomright']['x']
        y2 = result['bottomright']['y']
        label = result['label']
        if label != 'person':
            continue
        confidence = result['confidence']
        str_out = '{}_{}'.format(label, confidence)
        cv2.rectangle(tmpi,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      3)

        cv2.putText(tmpi,
                    str_out,
                    (int(x1 + (x2-x1)/2), y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)
    return tmpi


def main_function(cam_url, image_url, queue_reader, area):
    '''
    This is main function
    '''
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))
    # Variables for tracking faces
    frame_counter = 0

    # Variables holding the correlation trackers and the name per faceid
    list_of_trackers = TrackersList()

    clear_tracking_folder()

    # Model for human detection
    print('Load YOLO model ...')
    options = {"model": "./cfg/yolo.cfg", "load": "../models/yolo.weights", "threshold": 0.5}
    detector = TFNet(options)

    # Model for person re-id
    body_extractor = BodyExtractor()

    if image_url is not None:
        imgcv = cv2.imread(image_url)
        results = detector.return_predict(imgcv)
        print(results)
        imgcv = draw_results(imgcv, results)
        print('Result drawn as ../test-data/result.jpg')
        cv2.imwrite('../test-data/result.jpg', imgcv)

    track_results = TrackerResultsDict()
    predict_dict = {}
    if Config.CALC_FPS:
        start_time = time.time()
    if args.cam_url is not None:
        frame_reader = URLFrameReader(args.cam_url, scale_factor=1)
    elif queue_reader is not None:
        frame_reader = RabbitFrameReader(rabbit_mq, queue_reader)
    else:
        print('Empty Image Source')
        return -1

    video_out_fps, video_out_w, video_out_h, = frame_reader.get_info()
    print(video_out_fps, video_out_w, video_out_h)
    center = (int(video_out_w/2), int(video_out_h/2))
    bbox = [int(center[0]-Config.Frame.ROI_CROP[0]*video_out_w),
            int(center[1]-video_out_h*Config.Frame.ROI_CROP[1]),
            int(center[0]+Config.Frame.ROI_CROP[2]*video_out_w),
            int(center[1]+Config.Frame.ROI_CROP[3]*video_out_h)]
    video_out = None

    if Config.Track.TRACKING_VIDEO_OUT:
        # new_w = abs(bbox[0] - bbox[2])
        # new_h = abs(bbox[1] - bbox[3])
        # print(new_w, new_h)
        video_out = VideoHandle('../data/tracking_video_out.avi',
                                video_out_fps,
                                int(video_out_w),
                                int(video_out_h))
    dlib_c_tr_video = cv2.VideoWriter('../data/check_tracking_video.avi',
                                      cv2.VideoWriter_fourcc(*'XVID'),
                                      video_out_fps,
                                      (int(video_out_w), int(video_out_h)))

    try:
        while True:  # frame_reader.has_next():
            frame = frame_reader.next_frame()
            if frame is None:
                print("Waiting for the new image")
                trackers_return_dict, predict_trackers_dict = \
                    list_of_trackers.check_delete_trackers(None, rabbit_mq)
                track_results.update_two_dict(trackers_return_dict)
                predict_dict.update(predict_trackers_dict)
                # list_of_trackers.trackers_history.check_time(None)
                # if args.cam_url is not None:
                #     frame_reader = URLFrameReader(args.cam_url, scale_factor=1)
                # elif queue_reader is not None:
                #     frame_reader = RabbitFrameReader(rabbit_mq, queue_reader)
                # else:
                #     print('Empty Image Source')
                #     return -1

                continue

            print("Frame ID: %d" % frame_counter)

            # frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

            if Config.Track.TRACKING_VIDEO_OUT:
                video_out.tmp_video_out(frame)
            if Config.CALC_FPS:
                fps_counter = time.time()

            tmpi = list_of_trackers.update_dlib_trackers(frame)
            dlib_c_tr_video.write(tmpi)

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                start_time = time.time()
                detected_bbs = []
                results = detector.return_predict(frame)
                print('Detect time: {}'.format(1/(time.time() - start_time)))
                for result in results:
                    if result['label'] != 'person':
                        continue
                    x1 = result['topleft']['x']
                    y1 = result['topleft']['y']
                    x2 = result['bottomright']['x']
                    y2 = result['bottomright']['y']

                    origin_bb = [x1, y1, x2, y2]
                    detected_bbs.append(origin_bb)

                    print(is_inner_bb(bbox, origin_bb))
                    if not is_inner_bb(bbox, origin_bb):
                        continue
                    if calc_bb_size(origin_bb) < 10000:
                        continue
                    bb_size = calc_bb_size(origin_bb)
                    body_image = frame[origin_bb[1]:origin_bb[3],
                                       origin_bb[0]:origin_bb[2], :]
                    # body_emb = body_extractor.extract_feature(body_image)

                    # TODO: refractor matching_detected_face_with_trackers
                    matched_fid = list_of_trackers.matching_face_with_trackers(frame,
                                                                               frame_counter,
                                                                               origin_bb,
                                                                               None,
                                                                               body_image,
                                                                               body_extractor)

                    # Update list_of_trackers
                    list_of_trackers.update_trackers_list(matched_fid,
                                                          time.time(),
                                                          origin_bb,
                                                          body_image,
                                                          bb_size,
                                                          area,
                                                          frame_counter,
                                                          None,
                                                          body_extractor,
                                                          rabbit_mq)

            # list_of_trackers.update_trackers_by_tracking(body_extractor,
            #                                              frame,
            #                                              area,
            #                                              frame_counter,
            #                                              detected_bbs,
            #                                              rabbit_mq)

            trackers_return_dict, predict_trackers_dict = \
                list_of_trackers.check_delete_trackers(None, rabbit_mq)
            track_results.update_two_dict(trackers_return_dict)
            predict_dict.update(predict_trackers_dict)

            # Check extract trackers history time (str(frame_counter) + '_' + str(i))
            # list_of_trackers.trackers_history.check_time(None)

            frame_counter += 1
            if Config.CALC_FPS:
                print("FPS: %f" % (1/(time.time() - fps_counter)))
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
        if Config.Track.PREDICT_DICT_OUT:
            PickleUtils.save_pickle(Config.PREDICTION_DICT_FILE, predict_dict)
    except KeyboardInterrupt:
        print('Keyboard Interrupt !!! Release All !!!')
        # list_of_trackers.trackers_history.check_time(matcher)
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format((frame_counter + 1)/(time.time() - start_time)))
        frame_reader.release()
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
            video_out.release()
        if Config.Track.PREDICT_DICT_OUT:
            PickleUtils.save_pickle(Config.PREDICTION_DICT_FILE, predict_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('For demo only',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',
                        '--cam_url',
                        help='your camera ip address',
                        default=None)
    parser.add_argument('-i',
                        '--image',
                        help='your image',
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
    parser.add_argument('-db',
                        '--dashboard',
                        help='Send dashboard result',
                        default=False)
    parser.add_argument('-mo',
                        '--min_dist_out',
                        help='write tracking folder with good element n min distance',
                        default=False)
    parser.add_argument('-cq',
                        '--queue_reader',
                        help='read frame from queue',
                        default=None)
    args = parser.parse_args()

    # Run
    if args.write_images == 'True':
        Config.Track.FACE_TRACK_IMAGES_OUT = True
    if args.video_out == 'True':
        Config.Track.TRACKING_VIDEO_OUT = True
    if args.dashboard == 'True':
        Config.SEND_QUEUE_TO_DASHBOARD = True
    if args.min_dist_out == 'True':
        Config.Track.MIN_MATCH_DISTACE_OUT = True
    Config.Track.PREDICT_DICT_OUT = True

    main_function(args.cam_url, args.image, args.queue_reader, args.area)
