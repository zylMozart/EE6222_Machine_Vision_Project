import mmcv
import numpy as np

if __name__=='__main__':
    ann_file='data/ntu_dark_skeletions/train/ntu_dark_skeleton_train.pkl'
    data = mmcv.load(ann_file)
    for item in data:
        print(item['keypoint_score'].shape)
        print(np.mean(item['keypoint_score']))
    pass