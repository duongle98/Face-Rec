import os
import requests
import argparse


def main(api_dir):
    if not os.path.exists(api_dir):
        raise ValueError('API folder does not exists')
    update_dict = {}
    with open(os.path.join(api_dir, 'update_token.txt'), 'r') as update_file:
        for line in update_file:
            line = line.strip().split()
            update_dict[line[0]] = line[1]

    i = 1
    for token, face_id in update_dict.items():
        data = {'face_id': face_id}
        request_url = "http://210.211.119.152:5000/update/" + token
        response = requests.post(request_url, json=data)
        assert response.json() == {"result": "ok"}, "Update failed"
        print("Test {} passed".format(i))
        i += 1


def parse_arguments():
    parser = argparse.ArgumentParser('parser for api update test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--api_dir', default='../../data/api_testing')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments().api_dir)
