import sys
import cv2

import license_plate
import feature_matching


if __name__ == "__main__":
    if len(sys.argv) == 2:
        license_plate.read_license_plate(sys.argv[1])
    elif len(sys.argv) == 3:
        output, match = None, None
        query_image = cv2.imread(sys.argv[1])
        train_image = cv2.imread(sys.argv[2])

        train, train_match = feature_matching.keypoint_matching(query_image, train_image)
        query, query_match = feature_matching.keypoint_matching(train_image, query_image)

        if len(query_match) > len(train_match):
            output, match = query, query_match
        else:
            output, match = train, train_match
        print("Number of matches: %i" % (len(match)))

        if len(match) >= 38:
            print("Images match")
        else:
            print("Images do not match")

    pass
