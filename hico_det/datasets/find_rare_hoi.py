"""
Created on Oct 22, 2017

@author: Siyuan Qi

Description of the file.

"""

import os

import numpy as np
import scipy.io



def collect_hoi_stats(bbox):
    stats = np.zeros(600)
    for idx in range(bbox['filename'].shape[1]):
        for i_hoi in range(bbox['hoi'][0, idx]['id'].shape[1]):
            hoi_id = bbox['hoi'][0, idx]['id'][0, i_hoi][0, 0]
            stats[int(hoi_id)-1] += 1
    print(stats)

    return stats


def split_testing_set(paths, bbox, stats):
    # feature_path = os.path.join(paths, 'processed', 'features_background_49')

    rare_set = list()
    non_rare_set = list()
    for idx in range(bbox['filename'].shape[1]):
        filename = str(bbox['filename'][0, idx][0])
        filename = os.path.splitext(filename)[0] + '\n'
        #
        # try:
        #     det_classes = np.load(os.path.join(feature_path, '{}_classes.npy'.format(filename.strip())))
        # except IOError:
        #     continue

        rare = False
        for i_hoi in range(bbox['hoi'][0, idx]['id'].shape[1]):
            hoi_id = bbox['hoi'][0, idx]['id'][0, i_hoi][0, 0]

            if stats[int(hoi_id)-1] < 10:
                rare_set.append(filename)
                rare = True
                continue
        if not rare:
            non_rare_set.append(filename)

    with open('./test_rare.txt', 'w') as f:
        f.writelines(rare_set)

    with open('./test_non_rare.txt', 'w') as f:
        f.writelines(non_rare_set)


def find_rare_hoi(paths):
    anno_bbox = scipy.io.loadmat(os.path.join(paths, 'anno_bbox.mat'))
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']
    list_action = anno_bbox['list_action']

    stats = collect_hoi_stats(bbox_test)
    split_testing_set(paths, bbox_test, stats)


def main():
    # paths = hico_config.Paths()
    paths='/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det'
    find_rare_hoi(paths)


if __name__ == '__main__':
    main()
