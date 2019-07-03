import sys

import numpy as np
import cv2
from matplotlib import pyplot as plt


def keypoint_matching(query_image, train_image):
    MIN_MATCH_COUNT = 38
    FLANN_INDEX_KDTREE = 0
    THRESHOLD = 0.8

    # initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoint and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_image, None)
    kp2, des2 = sift.detectAndCompute(train_image, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < THRESHOLD*n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w, _ = query_image.shape
        points = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(points, M)
        train_image = cv2.polylines(train_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        matches_mask = None

    draw_params = dict(
        matchColor=(0, 255, 0), singlePointColor=None,
        matchesMask=matches_mask, flags=2
    )
    output = cv2.drawMatches(query_image, kp1, train_image, kp2, good_matches, None, **draw_params)
    return output, good_matches


if __name__ == "__main__":
    output = keypoint_matching(sys.argv[1], sys.argv[2])
    plt.imshow(output)
    plt.show()
