import rethinkdb as r
from threading import Thread
import time
import requests
import os
import shutil
import argparse


table = {}


def populate_table(table_name, records):
    conn = r.connect('210.211.119.152', 28015, user='tch', password='tch', timeout=3600)
    cursor = r.db('TCH').table(table_name).changes().run(conn)
    idx = 0
    for change in cursor:
        print("There are changed in main table")
        records[idx] = change['new_val']
        idx += 1


def insert_to_clone(table_name, records):
    conn = r.connect('210.211.119.152', 28015, user='admin', password='admin', timeout=3600)
    prev_id = -2
    while True:
        time.sleep(2)
        if len(records) > prev_id + 1 and len(records) > 0:
            r.db('TCH').table(table_name).insert(records[len(records) - 1]).run(conn)
            print("New record is inserted to clone table")
            prev_id = len(records) - 1


def save_image(url, name, dir):
    r = requests.get(url, stream=True)
    with open(os.path.join(dir, name), 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)


def listen_to_clone(table_name, image_dir):
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    hit_times = 0
    conn = r.connect('210.211.119.152', 28015, user='admin', password='admin', timeout=3600)
    cursor = r.db('TCH').table(table_name).changes().run(conn)
    for change in cursor:
        change = change['new_val']
        if hit_times < 5:
            token = change['token']
            api_request = "http://210.211.119.152:5000/retrieve/" + token
            api_response = requests.get(api_request)
            if change['visitor_type'] == 'old':
                try:
                    os.mkdir(os.path.join(image_dir, token))
                except FileExistsError:
                    continue
                capture_img_url = "http://210.211.119.152:5000/" + change['captured']
                print(capture_img_url)
                save_image(capture_img_url, 'captured.jpg', os.path.join(image_dir, token))
                for match in change['matched']:
                    if 'NOT_FOUND' not in match['image']:
                        match_id = match['face_id']
                        match_img_url = "http://210.211.119.152:5000/" + match['image']
                        print(match_img_url)
                        save_image(match_img_url, match_id + '.jpg', os.path.join(image_dir, token))

            assert api_response.status_code == 200, "API not return status 200"
            print("Test {} passed".format(hit_times + 1))
            hit_times += 1
        else:
            break


def main(api_dir):
    if not os.path.exists(api_dir):
        raise ValueError('API folder does not exists')

    print("Begin to test")
    conn = r.connect('210.211.119.152', 28015, user='admin', password='admin', timeout=3600)
    table_name = "clone_table"
    r.db('TCH').table_create(table_name).run(conn)

    t1 = Thread(target=populate_table, args=('data', table, ))
    t1.daemon = True

    t2 = Thread(target=insert_to_clone, args=(table_name, table, ))
    t2.daemon = True

    t3 = Thread(target=listen_to_clone, args=(table_name, api_dir, ))

    t1.start()
    t2.start()
    t3.start()
    try:
        while(True):
            time.sleep(3)
            if not t3.isAlive():
                break
    except Exception:
        pass
    r.db('TCH').table_drop(table_name).run(conn)


def parse_arguments():
    parser = argparse.ArgumentParser('parser for api update test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--api_dir', default='../../data/api_testing')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments().api_dir)
