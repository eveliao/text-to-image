import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

# DID NOT TRAIN IT ON MS COCO YET
def save_caption_vectors_ms_coco(data_dir, split, batch_size):
    meta_data = {}
    ic_file = join(data_dir, 'annotations/captions_{}2014.json'.format(split))
    with open(ic_file) as f:
        ic_data = json.loads(f.read())

    meta_data['data_length'] = len(ic_data['annotations'])
    with open(join(data_dir, 'meta_{}.pkl'.format(split)), 'wb') as f:
        pickle.dump(meta_data, f)

    model = skipthoughts.load_model()
    batch_no = 0
    print "Total Batches", len(ic_data['annotations'])/batch_size

    while batch_no*batch_size < len(ic_data['annotations']):
        captions = []
        image_ids = []
        idx = batch_no
        for i in range(batch_no*batch_size, (batch_no+1)*batch_size):
            idx = i%len(ic_data['annotations'])
            captions.append(ic_data['annotations'][idx]['caption'])
            image_ids.append(ic_data['annotations'][idx]['image_id'])

        print captions
        print image_ids
        # Thought Vectors
        tv_batch = skipthoughts.encode(model, captions)
        h5f_tv_batch = h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no)), 'w')
        h5f_tv_batch.create_dataset('tv', data=tv_batch)
        h5f_tv_batch.close()

        h5f_tv_batch_image_ids = h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no)), 'w')
        h5f_tv_batch_image_ids.create_dataset('tv', data=image_ids)
        h5f_tv_batch_image_ids.close()

        print "Batches Done", batch_no, len(ic_data['annotations'])/batch_size
        batch_no += 1


def save_caption_vectors_flowers(data_dir):
    import time

    img_dir = join(data_dir, 'flowers')
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    # print image_files[300:400]
    print len(image_files)
    print (image_files)
    for img_file in image_files:
        print (img_file)
        print type(img_file)
    image_captions = { img_file : [] for img_file in image_files }

    caption_dir = join(data_dir, 'flowers/text_c10')
    class_dirs = []
    for i in range(1, 103):
        class_dir_name = 'class_%.5d'%(i)
        class_dirs.append( join(caption_dir, class_dir_name))

    for class_dir in class_dirs:
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(join(class_dir,cap_file)) as f:
                captions = f.read().split('\n')
            img_file = cap_file[0:11] + ".jpg"
            # 5 captions per image
            image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

    print len(image_captions)

    model = skipthoughts.load_model()
    encoded_captions = {}


    for i, img in enumerate(image_captions):
        st = time.time()
        encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
        print i, len(image_captions), img
        print "Seconds", time.time() - st


    h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
    for key in encoded_captions:
        h.create_dataset(key, data=encoded_captions[key])
    h.close()



def save_caption_vectors_traffic(data_dir):
    import time
    import psycopg2
    hostname = '202.120.38.146'
    username = 'icdm'
    password = 'icdm2017'
    database = 'nyctraffic'
    port = 9432

    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database, port=port)
    conn.set_session(autocommit=True)


    cur = conn.cursor()
    sql = "select * from links_with_feature"
    cur.execute(sql)
    spatial_feature = dict()

    for row in cur.fetchall():
        tmp = []
        link_id = row[0]
        for i in range(3, len(row) - 1):
            tmp.append(row[i])
        spatial_feature[link_id] = tmp



    h = h5py.File(join(data_dir, 'traffic_tv.hdf5'))
    for key in spatial_feature:
        h.create_dataset(key, data=spatial_feature[key])
    h.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                       help='train/val')
    parser.add_argument('--data_dir', type=str, default='/home/eve/traffic/text2img/Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    parser.add_argument('--data_set', type=str, default='flowers',
                       help='Data Set : Flowers, MS-COCO')
    args = parser.parse_args()

    if args.data_set == 'flowers':
        save_caption_vectors_traffic(args.data_dir)
    else:
        save_caption_vectors_ms_coco(args.data_dir, args.split, args.batch_size)

if __name__ == '__main__':
    main()