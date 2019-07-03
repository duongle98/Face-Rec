import os
import pickle
import datetime

import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils


# initialize dlib face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load classifiers
classifier_glass_dir = os.path.expanduser("fraud_detection/model_glass.pkl")
with open(classifier_glass_dir, "rb") as infile:
    classifier_glass = pickle.load(infile)

classifier_mask_dir = os.path.expanduser("fraud_detection/model_mask.pkl")
with open(classifier_mask_dir, "rb") as infile:
    classifier_mask = pickle.load(infile)
    print(type(classifier_mask))


def normalize_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    threshold = 100
    v[(v >= (threshold-30)) & (v < threshold)] = 100
    v[v < threshold] += 30
    hsv_image = cv2.merge((h, s, v))
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image


def detect_glasses(image, face, shape):
    # left eye
    (x, y, w, h) = cv2.boundingRect(np.array([shape[42:48]]))
    left_eye = cv2.resize(image[y:y+h, x:x+w], (20, 7))
    # right eye
    (x, y, w, h) = cv2.boundingRect(np.array([shape[36:42]]))
    right_eye = cv2.resize(image[y:y+h, x:x+w], (20, 7))
    # left eyebrow
    (x, y, w, h) = cv2.boundingRect(np.array([shape[22:27]]))
    left_eyebrow = cv2.resize(image[y:y+h, x:x+w], (40, 15))
    # right eyebrow
    (x, y, w, h) = cv2.boundingRect(np.array([shape[17:22]]))
    right_eyebrow = cv2.resize(image[y:y+h, x:x+w], (40, 15))

    # glass feature
    eyes = np.concatenate((left_eye, right_eye), 1)
    glass_feature = np.concatenate((eyes, left_eyebrow, right_eyebrow))
    out = classifier_glass.predict([glass_feature.flatten()])[0]
    return out


def detect_mask(image, face, shape):
    # nose
    (x, y, w, h) = cv2.boundingRect(np.array([shape[27:36]]))
    nose = cv2.resize(image[y:y+h, x:x+w], (25, 40))
    # transpose nose
    nose = np.transpose(nose, (1, 0, 2))
    # mouth
    (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
    mouth = cv2.resize(image[y:y+h, x:x+w], (40, 25))

    # mask feature
    mask_feature = np.concatenate((nose, mouth))
    out = classifier_mask.predict([mask_feature.flatten()])[0]
    return out


def detect_parts(image, face):
    # detect facial landmarks for face region, convert to a NumPy array
    shape = predictor(image, face)
    shape = face_utils.shape_to_np(shape)
    parts = []

    result_glasses = detect_glasses(image, face, shape)
    if result_glasses == "glasses":
        parts.append("glasses")

    result_mask = detect_mask(image, face, shape)
    if result_mask == "mask":
        parts.append("mask")

    if not parts:
        parts.append("clear")
    cv2.putText(
        image, ", ".join(parts), (face.left(), face.top()-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    return image


def main():
    cap = cv2.VideoCapture(-1)

    while True:
        res, image = cap.read()
        if not res:
            continue
        image = imutils.resize(image, width=400)

        # normalize brightness
        image = normalize_brightness(image)
        faces = detector(image)

        if len(faces) == 0:
            cv2.putText(
                image, "your face is unclear", (2, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("faces", image)

            if cv2.waitKey(100) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
            continue

        for face in faces:
            cv2.rectangle(
                image, (face.left(), face.top()), (face.right(), face.bottom()),
                (255, 0, 255), 1, cv2.LINE_AA)
            image = detect_parts(image, face)
        cv2.imshow("faces", image)

        # show result
        time = datetime.datetime.now()
        print(time)

        if cv2.waitKey(100) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
