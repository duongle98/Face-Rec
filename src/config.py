"""
All eyeq configuration go here
"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(current_dir, os.path.pardir))
MAIN_NAME = '30shine'
ENV_MODE = 'production'

class Config:
    '''
    Contain common configuration
    '''

    # class Mode: TODO: Refractor
    LOG = True
    QA_MODE = False
    SHOW_FRAME = True
    DEBUG = True
    SEND_RBMQ_HTTP = 'http://cdn.eyeq.tech:1111/30shine_images'
    QUERY_TOP10_MODE = False
    MAIN_NAME = MAIN_NAME

    SEND_QUEUE_TO_DASHBOARD = False
    CALC_FPS = True
    TIME_KILL_NON_ACTIVE_PROCESS = 5.0
    FRAME_QUEUE_SIZE = 200
    STREAM_TIMEOUT = 10
    FRAME_ON_DISK = "../data/stream"
    MAX_IMAGES_PER_TRACK = 100
    NUM_IMAGE_PER_EXTRACT = 5
    
    class Track:
        '''
        Configuration for tracking
        '''
        INIT_FACE_ID = 'DETECTING'
        TRACKER = 'KCF'
        BAD_TRACK = 'BAD-TRACK'
        TRACKING_VIDEO_OUT = False
        TRACKING_QUEUE_CAM_TO_CENTRAL = False
        SEND_RECOG_API = False
        RECOG_TIMER = 30
        MIN_OVERLAP_IOU = 0.3
        MIN_NOF_TRACKED_FACES = 15
        RECOGNIZE_FULL_TRACK = True
        CURRENT_EXTRACR_TIMER = 15
        TRACK_TIMEOUT = 20
        LONGTTERM_HISTORY_TIMER = 200
        SHORTTERM_HISTORY_TIMER = 100
        HISTORY_EXTRACT_TIMER = 400
        RETRACK_THRESHOLD = 0.7
        RETRACK_MINRATE = 0.3
        VALID_ELEMENT_MINRATE = 0.8
        COMBINED_MATCHER_RATE = 0.7
        DLIB_TRACK_QUALITY = 6
        HISTORY_RETRACK_THRESHOLD = 0.66
        HISTORY_RETRACK_MINRATE = 0.5
        NOF_REGISTER_IMAGES = 100
        SKIP_FRAME = 15
        VIDEO_OUT_PATH = '/mnt/data/TCH_data'
        BB_SIZE = 0

    # GPU Config
    GPU_DEVICE = '/gpu:0'
    GPU_FRACTION = 1

    class Rabbit:
        '''
        Configuration for sending message via rabbitmq
        '''
        USERNAME = 'admin'
        PASSWORD = 'admin123'
        IP_ADDRESS = 'localhost'
        PORT = 5672
        INTER_SEP = '|'
        INTRA_SEP = '?'
    
    class Queues:
        MERGE = '-'.join([MAIN_NAME, 'merge', ENV_MODE])
        SPLIT = '-'.join([MAIN_NAME, 'split', ENV_MODE])
        ACTION = '-'.join([MAIN_NAME, 'action', ENV_MODE])
        ACTION_RESULT = '-'.join([MAIN_NAME, 'action', 'result', ENV_MODE])
        LIVE_RESULT = '-'.join([MAIN_NAME, ENV_MODE])

    class Filters:
        YAW = 60
        PITCH = 60
        COEFF = 0.11

    class MongoDB:
        '''
        Configuration for MongoDB
        '''
        USERNAME = ''
        PASSWORD = ''
        IP_ADDRESS = 'localhost'
        PORT = 27017
        DB_NAME = '-'.join([MAIN_NAME, 'cv'])
        DASHINFO_COLS_NAME = 'dashinfo'
        FACEINFO_COLS_NAME = 'faceinfo'
        MSLOG_COLS_NAME = 'mslog'

    class Frame:
        '''
        Configuration for reading frame
        '''
        FRAME_INTERVAL = 1  # read 1 every x frames
        SCALE_FACCTOR = 1  # rescale frame for faster face detection
        ROI_CROP = (0, 0, 1, 1)

    class MTCNN:
        '''
        Configuration for mtcnn face detector
        '''
        MIN_FACE_SIZE = 100  # minimum size of face
        THRESHOLD = [0.6, 0.7, 0.7]  # three steps's threshold
        FACTOR = 0.709  # default 0.709 image pyramid -- magic number
        SCALE_FACTOR = 3
    class Align:
        '''
        Configuration to align faces
        '''
        MARGIN = 32
        IMAGE_SIZE = 160
        ROI_RATE_W = 110
        ROI_RATE_H = 140
        UPPER_RATE = 0.8
        LOWER_RATE = 1.2
        ASPECT_RATIO = 1.5
        MAX_FACE_ANGLE = 50

    class Matcher:
        '''
        Configuration for finding matched id via kd-tree
        '''
        RETRIEVAL_MODEL = 'faiss'
        CLOSE_SET_SVM = False
        MAX_TOP_MATCHES = 100
        INDEXING_INTERVAL = 30 * 60  # how often the matcher will update itself, in seconds
        INDEX_LEAF_SIZE = 2
        EMB_LENGTH = 128
        MIN_ASSIGN_THRESHOLD = 0.66
        MIN_ASSIGN_RATE = 0.5
        NON_FACE = 'NON_FACE'
        NEW_FACE = 'NEW_FACE'
        EXCESS_NEW_REGISTER = 150
        CLEAR_SESSION = False
        ADD_NEW_FACE = True
        USE_IMAGE_ID = False

    class RPCServer:
        DST_PATH = r'/home/vtdc/eyeq-card-face-mapping-api'
        SAVE_PATH = r'public/results'
    # eyeq trained model
    classifier_filename = '%s/models/users.pkl' % ROOT
    embedding_filename = '%s/models/embeddings.pkl' % ROOT

    # default extension
    IMG_EXT = 'jpg'
    PKL_EXT = 'pkl'
    LOCAL_CAM_ID = 'WEB_SERVER'

    # path
    FACENET_DIR = '%s/models/coefficient_resnet_v1.pb' % ROOT
    COEFF_DIR = '%s/models/coefficient_resnet_v1.pb' % ROOT
    MTCNN_DIR = '%s/models/align' % ROOT
    DEBUG_PATH = '%s/data' % ROOT
    HAAR_FRONTAL = '%s/models/haarcascade_frontalface_default.xml' % ROOT
    """ DB SETUP, SHARED BETWEEN CENTRAL WORKER AND MATCHING WORKER"""
    # class Dir:
    SESSION_DIR = '%s/session' % ROOT
    DATA_DIR = '%s/data' % ROOT
    TRACKING_DIR = os.path.join(DATA_DIR, 'tracking')
    # send rbmq dir
    SEND_RBMQ_DIR = os.path.join(DATA_DIR, 'send_rbmq')
    # contain dashboard images
    DB_IMAGES_DIR = os.path.join(DATA_DIR, 'dashboard_images')
    # db contains the main register data, kdtree as pkl10
    DB_DIR = os.path.join(SESSION_DIR, 'db')
    # live save original frame as format AREA_TIME_BB.pkl
    LIVE_DIR = os.path.join(SESSION_DIR, 'live')
    # reg contain register image for each  person
    REG_DIR = os.path.join(SESSION_DIR, 'reg')
    # frames cotains all frame needed for qa
    FRAME_DIR = os.path.join(SESSION_DIR, 'frames')

    """ pkl file """
    # class File:
    # a data structure for registered [image_id: face_id]
    REG_IMAGE_FACE_DICT_FILE = os.path.join(DB_DIR, 'reg_image_face_dict.pkl')
    # a data structure to log changes
    FACE_CHANGES_FILE = os.path.join(DB_DIR, 'face_changes_list.pkl')
    # a data structure for trained kdtree (timestamp, tree, len, image_ids)
    MATCHER_TUP_FILE = os.path.join(SESSION_DIR, 'matcher_tup.pkl')
    # a data structure for trained svm (timestamp, svm, len)
    KDTREE_MATCHER_TUP_FILE = os.path.join(DB_DIR, 'kdtree_matcher_tup.pkl')
    FAISS_MATCHER_TUP_FILE = os.path.join(DB_DIR, 'faiss_matcher_tup.pkl')
    # a data structure for traiend kd tree (timestamp, tree, len, image_id)
    SVM_MATCHER_TUP_FILE = os.path.join(DB_DIR, 'svm_matcher_tup.pkl')
    # a data structure for traiend kd tree (last_time, embs, length)
    LINEAR_MATCHER_TUP_FILE = os.path.join(DB_DIR, 'linear_matcher_tup.pkl')
    # store the predict id of detected face from live run
    LIVE_DICT_FILE = os.path.join(DB_DIR, 'live_dict.pkl')
    # keep info for each frame as 22
    FRAME_IMAGE_DICT_FILE = os.path.join(DB_DIR, 'frames.pkl')
    # run QA mode to generate this to compare against groundtruth, live_dict.pkl
    PREDICTION_DICT_FILE = os.path.join(DB_DIR, 'prediction.pkl')
