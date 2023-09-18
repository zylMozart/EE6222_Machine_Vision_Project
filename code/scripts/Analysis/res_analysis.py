import numpy as np
import matplotlib.pyplot as plt
import re
import json
import pandas as pd

json_list=[
# Compare with baseline
# 'work_dirs/slowonly_r50_4x16x1_256e_kinetics400_rgb/20221016_215743.log.json',
'work_dirs/stgcn_80e_ntu60_xsub_keypoint_ntu/20221018_145733.log.json',
'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID/20221019_171721.log.json',
# 'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/20221019_165906.log.json',
'work_dirs/skeleton3dnew/GID_linear_x/20221026_115228.log.json',
'work_dirs/skeleton3dmore/GID_0.2_0.5_good_linear_x/20221028_103536.log.json',
'work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.02/20221028_220945.log.json',
# 'work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.2/20221028_220958.log.json',
# 'work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.02/20221028_220945.log.json',
# 'work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.002/20221028_220933.log.json',

# Ablation
# 'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID/20221019_171721.log.json',
# 'work_dirs/skeleton3dnew/GID_linear_x/20221026_115228.log.json',
# 'work_dirs/skeleton3dmore/GID_0.2_0.5_good_linear_x/20221028_103536.log.json',

]

log_list=[
# For skeleton3d method
# 'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint2/20221019_024734.log',
# 'work_dirs/posec3d/skeleton3d_linear_x/20221023_205745.log',
# 'work_dirs/posec3d/skeleton3d_linear_relu_x/20221023_205659.log',
# 'work_dirs/posec3d/skeleton3d_linear_relu/20221023_205522.log',

# For skeleton3d with Enhancement
# 'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID/20221019_171721.log',
# 'work_dirs/posec3d/skeleton3d_None/20221025_152223.log',
# 'work_dirs/posec3d/skeleton3d_linear_rl_GID/20221024_135422.log',
# 'work_dirs/posec3d/skeleton3d_linear_rl_x_GID/20221024_135609.log',
# 'work_dirs/posec3d/skeleton3d_linear_GID/20221024_215758.log',
# 'work_dirs/posec3d/skeleton3d_linear_x_GID/20221024_215702.log'

# Update
'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID/20221019_171721.log',
'work_dirs/posec3d/skeleton3d_None/20221025_152223.log',
'work_dirs/skeleton3dnew/GID_linear_rl/20221026_115145.log',
'work_dirs/skeleton3dnew/GID_linear_x/20221026_115228.log',
'work_dirs/skeleton3dnew/GID_linear_x_rl/20221026_115048.log',
'work_dirs/skeleton3dmore/GID_0.2_0.5_linear_x/20221027_191436.log',
'work_dirs/skeleton3dmore/GID_0.2_0.5_good_linear_x/20221028_103536.log',
'work_dirs/skeleton3dmore/GID_0.2_linear_x/20221027_191125.log',
'work_dirs/skeleton3dmore/GID_0.3_0.3_linear_x/20221028_025357.log',

# For enhancement
# log_list=['work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint2/20221019_024734.log',
# 'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID/20221019_171721.log',
# 'work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_DCE/20221019_171730.log',
]

def plot_acc_change():
    file_list=json_list
    # file_list=log_list
    fig, ax1 = plt.subplots(1, 1, figsize=(15,3))
    for log in file_list:
        data=json2indicator(log)
        # data=log2acc(log)
        ax1.plot(data['epoch'],data['top1acc'],label='Sk'+log.split('/')[-2][-10:])
        ax1.grid(True)
        # print('Best Acc1 of {}:{}'.format(log.split('/')[-2][-15:],np.max(data['top1acc'])))
        max_i = np.argmax(data['top1acc'])
        print(r'\hline')
        print('{} & {:.4f} & {:.4f} \\\\'.format(log.split('/')[-2][-15:],data['top1acc'][max_i],data['top5acc'][max_i]))
    plt.legend()
    plt.savefig('scripts/Analysis/files/acc_compare.png')
    pass

def log2acc(filepath):
    nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
    with open(filepath,'r') as f:
        content=f.readlines()
    res={'epoch':[],'top1acc':[],'top5acc':[],'meanacc':[]}
    for item in content:
        if 'mmaction - INFO - Epoch(val)' in item:
            epoch=int(re.findall('Epoch\(val\) \[(.*)\]\[',item)[0])
            top1acc=float(re.findall('top1_acc: (.*), top5',item)[0])
            top5acc=float(re.findall('top5_acc: (.*), mean',item)[0])
            meanacc=float(re.findall('mean_class_accuracy: (.*)\n',item)[0])
            res['epoch'].append(epoch)
            res['top1acc'].append(top1acc)
            res['top5acc'].append(top5acc)
            res['meanacc'].append(meanacc)
    return res

def json2indicator(jsonpath):
    with open(jsonpath,'r') as f:
        content=f.readlines()
    res={'epoch':[],'top1acc':[],'top5acc':[],'meanacc':[]}
    for item in content[1:]:
        json_item=json.loads(item)
        if json_item['mode']=='val':
            res['epoch'].append(json_item['epoch'])
            res['top1acc'].append(json_item['top1_acc'])
            res['top5acc'].append(json_item['top5_acc'])
            # res['meanacc'].append(json_item['mean_class_accuracy'])
    return res

def dataset_statics():
    res={}
    with open('data/ntu_dark/ntu_dark_train_list.txt','r') as f:
        content = f.readlines()
    for item in content:
        label=item.split(' ')[-1]
        if label not in res.keys():
            res[label]=1
        else:
            res[label]+=1
    with open('data/ntu_dark/ntu_dark_test_list.txt','r') as f:
        content = f.readlines()
    for item in content:
        label=item.split(' ')[-1]
        if label not in res.keys():
            res[label]=1
        else:
            res[label]+=1
    pass

if __name__=='__main__':
    # json2indicator('work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.2/20221028_220958.log.json')
    plot_acc_change()
    # dataset_statics()
    ### plot_skeleton()
