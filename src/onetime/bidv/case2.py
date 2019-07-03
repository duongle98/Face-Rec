import time
import random

import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils


def angle_x(vector):
    return np.arccos(np.dot(vector / np.linalg.norm(vector), [1, 0])) / np.pi


def angle_y(vector):
    return np.arcsin(np.dot(vector / np.linalg.norm(vector), [0, 1])) / np.pi


def direction_x(vector):
    if angle_x(vector) <= 1/3:
        return "left"
    if angle_x(vector) >= 2/3:
        return "right"
    return "middle"


def direction_y(vector):
    if angle_y(vector) <= -0.15:
        return "up"
    if angle_y(vector) >= 0.15:
        return "down"
    return "middle"


def normalize_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    threshold = 100
    v[(v >= (threshold-30)) & (v < threshold)] = 100
    v[v < threshold] += 30
    hsv_image = cv2.merge((h, s, v))
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image


def generate_rules(keys, num_trials):
    # rules = random.choices(keys, k=num_trials)
    rules = []
    i = 0
    key_index, old_index = 0, 0
    while i <= num_trials:
        if i != 0:
            old_index = key_index
        while i != 0 and key_index == old_index:
            key_index = random.randrange(0, len(keys))

        rules.append(keys[key_index])
        i += 1
    return rules


def compute_direction(image, face, target_direction, data):
    [predictor, model_points, camera_matrix, dist_coeffs, directions] = data

    # detect facial landmarks
    shape = predictor(image, face)
    shape = face_utils.shape_to_np(shape)

    # detect key features
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left Mouth corner
        shape[54]   # Right mouth corner
    ], dtype=float)

    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix,
        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # project a 3D point (0, 0, 1000.0) onto the image plane.
    # we use this to draw a line sticking out of the nose
    (nose_end_point_2D, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
        translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point_2D[0][0][0]), int(nose_end_point_2D[0][0][1]))

    cv2.line(image, p1, p2, (255, 0, 0), 2)
    line = np.array(p2) - np.array(p1)

    # direction for verification
    if directions[target_direction]:
        direction = direction_x(line)
    else:
        direction = direction_y(line)
    return direction


def main():
    # initialize dlib face detector (HOG-based) and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # constant parameters
    keys = ["left", "right", "up", "down"]
    keys = ["left", "right", "up"]
    directions = {"left": True, "right": True, "up": False, "down": False}
    num_trials = 5
    trial_step = 4
    # time_out = 7

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left Mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner

    ])

    dist_coeffs = np.zeros((4, 1))
    # Camera internals
    size = [300, 400]
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=float
    )

    # verification parameters
    sum_step = 0
    cur_step = 0
    pointer = 0
    num_fail = 0
    first_time = True
    direction = ""
    cur_time = time.time()

    cap = cv2.VideoCapture(-1)
    verification_mode = False
    verification_result = 0
    rules = []

    while True:
        res, image = cap.read()
        if not res:
            continue
        image = imutils.resize(image, width=400)

        # detect faces
        faces = detector(image)
        if len(faces) == 0 or len(faces) == 2:
            reason = "many faces in camera" if len(faces) else "your face is unclear"
            cv2.putText(
                image, reason, (2, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("output", image)

            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
            continue

        # draw faces
        face = faces[0]
        cv2.rectangle(
            image, (face.left(), face.top()), (face.right(), face.bottom()),
            (255, 0, 255), 1, cv2.LINE_AA)

        # verification direction
        if verification_mode:
            direction = compute_direction(
                image, face, rules[pointer],
                [predictor, model_points, camera_matrix, dist_coeffs, directions])
            cv2.putText(
                image, "put your head " + rules[pointer], (2, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                image, "x-y directions %s" % direction, (2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # display image
        if verification_result == 1:
            cv2.putText(
                image, "verification succeeds", (200, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        elif verification_result == -1:
            cv2.putText(
                image, "verification fails", (200, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("output", image)

        # verification result of trial
        if verification_mode:
            if first_time:
                cur_time = time.time()
                first_time = False

            if time.time() - cur_time >= 2:
                if cur_step < trial_step:
                    if direction == rules[pointer]:
                        sum_step += 1
                    cur_step += 1
                else:
                    # determine if user follows instruction or not
                    if sum_step < 3:
                        num_fail += 1
                        print("fail")
                    else:
                        print("good")

                    # start new set of trials
                    cur_step = 0
                    sum_step = 0
                    pointer += 1
                    cur_time = time.time()

        # verification summary result
        if verification_mode and (pointer >= num_trials or num_fail > 1):
            if num_fail > 1:
                verification_result = -1
            else:
                verification_result = 1
            # reset parameters
            verification_mode = False
            sum_step = 0
            cur_step = 0
            pointer = 0
            num_fail = 0
            first_time = True
            direction = ""
            cur_time = time.time()

        # wait for key
        key = cv2.waitKey(100)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == 118:
            verification_mode = True
            rules = generate_rules(keys, num_trials)
            print("")

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
