import sys
import math
import re
import numpy as np
import cv2
from openalpr import Alpr


def init_detector():
    vn_detector = Alpr("vn", "openalpr.conf", "runtime_data")
    vn2_detector = Alpr("vn2", "openalpr.conf", "runtime_data")

    if not (vn_detector.is_loaded() or vn2_detector.is_loaded()):
        print("Error loading OpenALPR")
        raise Exception("Error loading OpenALPR")

    vn_detector.set_top_n(5)
    vn_detector.set_default_region("base")
    vn2_detector.set_top_n(5)
    vn2_detector.set_default_region("base")

    return vn_detector, vn2_detector


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def clear_newline(candidates, pattern):
    for candidate in candidates:
        candidate["plate"] = candidate["plate"].replace("\n", "")
        candidate["matches_template"] = bool(pattern.match(candidate["plate"]))
    return candidates


def read_license_plate(image_file):
    vn_detector, vn2_detector = init_detector()
    pattern = re.compile(r"^[0-9]{2}[A-Z][0-9]{5}$")

    results = vn2_detector.recognize_file(image_file)
    if len(results["results"]) == 0:
        results = vn_detector.recognize_file(image_file)

    if len(results["results"]) == 1:
        candidates = results["results"][0]["candidates"]
        return clear_newline(candidates, pattern)
    elif len(results["results"]) == 0:
        return "No license plate detected"

    image = cv2.imread(image_file)
    points = results["results"][0]["coordinates"]
    value = (points[3]["y"] - points[2]["y"]) / (points[2]["x"] - points[3]["x"])
    angle = int(math.degrees(math.atan(value)))

    rotated = rotate_image(image, -angle)
    image_file = "out.jpg"
    cv2.imwrite(image_file, rotated)

    results = vn2_detector.recognize_file(image_file)
    if len(results["results"]) < 2:
        new_results = vn_detector.recognize_file(image_file)
        if len(new_results["results"]) == 1:
            if len(results["results"]) == 1:
                return results["results"][0]["candidates"]
            return new_results["results"][0]["candidates"]
        elif len(new_results["results"]) == 0:
            if len(results["results"]) == 1:
                return results["results"][0]["candidates"]
            return "No license plate detected"
        results = new_results

    front_candidates = results["results"]["candidates"]
    back_candidates = results["results"]["candidates"]

    if back_candidates[0]["plate"][-2:] == front_candidates[0]["plate"][:2] \
            and front_candidates[0]["plate"].isdigit():
        front_candidates, back_candidates = back_candidates, front_candidates
    elif front_candidates[0]["plate"][-2:] != back_candidates[0]["plate"][:2]:
        return back_candidates

    front_pattern = re.compile(r"^[0-9]{2}[A-Z][0-9]{2}$")
    front_index = None
    for i, front_candidate in enumerate(front_candidates):
        if front_pattern.match(front_candidate["plate"]):
            front_index = i
            break

    if front_index is None:
        return back_candidates

    candidates = []
    for i in range(0, 5):
        front_candidate = front_candidates[front_index]
        back_candidate = back_candidates[i]

        plate_dict = dict()
        plate_dict["plate"] = front_candidate["plate"][:-2] + back_candidate["plate"]
        plate_dict["matches_template"] = bool(pattern.match(plate_dict["plate"]))
        plate_dict["confidence"] = \
            (front_candidate["confidence"] + back_candidate["confidence"]) / 2

        candidates.append(plate_dict)

    return candidates


if __name__ == "__main__":
    image_file = sys.argv[1]
    image = cv2.imread(image_file)

    result = read_license_plate(image_file)
    print(result)
