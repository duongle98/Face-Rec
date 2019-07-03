import urllib
import shutil

import cv2
from flask import request, jsonify
from flask_api import FlaskAPI

import license_plate
import feature_matching


app = FlaskAPI(__name__)


def response_success(result, message="success"):
    return jsonify({"status": "success", "message": message, "data": result})


def response_error_not_json():
    return jsonify({"status": "error", "message": "request is not json", "data": None})


def response_error_wrong_format():
    return jsonify({"status": "error", "message": "request is not in proper format", "data": None})


def response_error_missing_properties(properties):
    return jsonify({
        "status": "error", "message": "request is missing properties", "data": properties
    })


def download_url(url, local_path):
    with urllib.request.urlopen(url) as response, open(local_path, "wb+") as file:
        shutil.copyfileobj(response, file)
    return


def plate_equal(plate, target):
    if len(plate) == 5 and len(target) > 5:
        return plate == target[-5:]
    elif len(target) == 5 and len(plate) > 5:
        return plate[-5:] == target
    return plate == target


@app.route("/license-plate/numbers", methods=["GET", "POST"])
def read_license_plate():
    if not request.is_json:
        return response_error_not_json()
    if "image" not in request.json:
        return response_error_missing_properties(["image"])

    image_path = request.json["image"]
    local_path = "temp/01.jpg"

    download_url(image_path, local_path)
    results = license_plate.read_license_plate(local_path)

    return response_success(results)


@app.route("/license-plate/checking", methods=["GET", "POST"])
def checking_plate_number():
    if not request.is_json:
        return response_error_not_json()
    if ("image" not in request.json) or ("number" not in request.json):
        return response_error_missing_properties(["image", "number"])

    image_path = request.json["image"]
    number = request.json["number"]
    local_path = "temp/01.jpg"

    download_url(image_path, local_path)
    results = license_plate.read_license_plate(local_path)

    match = False
    for candidate in results:
        if plate_equal(candidate["plate"], number):
            match = True

    message = "the plate number is similar to %s" % (number) \
        if match else "the plate number is different"
    return response_success(number, message)


@app.route("/license-plate/matching", methods=["GET", "POST"])
def matching_plate_number():
    if not request.is_json:
        return response_error_not_json()
    if ("image_1" not in request.json) or ("image_2" not in request.json):
        return response_error_missing_properties(["image_1", "image_2"])

    image_path_1 = request.json["image_1"]
    image_path_2 = request.json["image_2"]
    local_path_1, local_path_2 = "temp/01.jpg", "temp/02.jpg"

    download_url(image_path_1, local_path_1)
    results_1 = license_plate.read_license_plate(local_path_1)
    download_url(image_path_2, local_path_2)
    results_2 = license_plate.read_license_plate(local_path_2)

    numbers = set()
    for candidate_1 in results_1:
        for candidate_2 in results_2:
            if plate_equal(candidate_1["plate"], candidate_2["plate"]):
                numbers.add(candidate_1["plate"])

    message = "the plate numbers are similar" if len(numbers) > 0 \
        else "the plate numbers are different"
    return response_success(str(numbers), message)


@app.route("/matching/ad", methods=["GET", "POST"])
def ad_matching():
    if not request.is_json:
        return response_error_not_json()
    if ("image_1" not in request.json) or ("image_2" not in request.json):
        return response_error_missing_properties(["image_1", "image_2"])

    THRESHOLD = 40

    image_path_1 = request.json["image_1"]
    image_path_2 = request.json["image_2"]
    local_path_1, local_path_2 = "temp/01.jpg", "temp/02.jpg"

    download_url(image_path_1, local_path_1)
    download_url(image_path_2, local_path_2)
    image_1 = cv2.imread(local_path_1)
    image_2 = cv2.imread(local_path_2)

    match = None, None
    _, train_match = feature_matching.keypoint_matching(image_1, image_2)
    _, query_match = feature_matching.keypoint_matching(image_2, image_1)

    if len(query_match) > len(train_match):
        match = query_match
    else:
        match = train_match

    message = "two ads are similar" if len(match) >= THRESHOLD \
        else "two ads are different"
    return response_success(len(match), message)


@app.route("/matching/image", methods=["GET", "POST"])
def image_matching():
    if not request.is_json:
        return response_error_not_json()
    if ("image_1" not in request.json) or ("image_2" not in request.json):
        return response_error_missing_properties(["image_1", "image_2"])

    THRESHOLD = 150

    image_path_1 = request.json["image_1"]
    image_path_2 = request.json["image_2"]
    local_path_1, local_path_2 = "temp/01.jpg", "temp/02.jpg"

    download_url(image_path_1, local_path_1)
    download_url(image_path_2, local_path_2)
    image_1 = cv2.imread(local_path_1)
    image_2 = cv2.imread(local_path_2)

    match = None
    _, train_match = feature_matching.keypoint_matching(image_1, image_2)
    _, query_match = feature_matching.keypoint_matching(image_2, image_1)

    if len(query_match) > len(train_match):
        match = query_match
    else:
        match = train_match

    message = "two images are similar" if len(match) >= THRESHOLD \
        else "two images are different"
    return response_success(len(match), message)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
    pass
