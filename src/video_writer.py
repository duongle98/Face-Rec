'''
Video Writer
'''
import os
import cv2


class VideoHandle:
    '''
    This class for writing output video
    '''
    def __init__(self, video_out_path, fps, w_video, h_video):
        # Define the codec and create VideoWriter object
        self.out_path = video_out_path
        self.tmp_path = '../data/tmp_video_for_tracking.avi'
        self.clear_video()
        self.out = cv2.VideoWriter(self.out_path,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   fps,
                                   (w_video, h_video))
        self.tmp_out = cv2.VideoWriter(self.tmp_path,
                                       cv2.VideoWriter_fourcc(*'XVID'),
                                       fps,
                                       (w_video, h_video))

    def write_track_video(self, track_results_dict):
        '''
        Write recognized tracking video
        '''
        print('Writing video ...')
        self.tmp_out.release()
        frame_reader = cv2.VideoCapture(self.tmp_path)
        frame_counter = 0
        while True:
            _, frame = frame_reader.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame is None:
                break
            if frame_counter in track_results_dict.keys():
                print('Write Frame: %d' % frame_counter)
                for i, name in enumerate(track_results_dict[frame_counter].track_names):
                    bb0 = int(track_results_dict[frame_counter].bounding_boxes[i][0])
                    bb1 = int(track_results_dict[frame_counter].bounding_boxes[i][1])
                    bb2 = int(track_results_dict[frame_counter].bounding_boxes[i][2])
                    bb3 = int(track_results_dict[frame_counter].bounding_boxes[i][3])
                    cv2.rectangle(frame, (bb0, bb1), (bb2, bb3), (0, 165, 255), 2)

                    cv2.putText(frame,
                                name,
                                (int(bb0+(bb2-bb0)/2), bb1),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2)
            self.out.write(frame)
            frame_counter += 1
        print('Video has been written as ' + self.out_path)
        # os.remove(self.tmp_path)

    def tmp_video_out(self, frame):
        '''
        Write temporary video
        '''
        self.tmp_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        '''
        Release all videohandles
        '''
        self.tmp_out.release()
        self.out.release()

    def clear_video(self):
        '''
        Remove tracking video and temporary video
        '''
        if os.path.exists(self.out_path):
            os.remove(self.out_path)
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
