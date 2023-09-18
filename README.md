# DSC3D
This is code implementation of our project "DKC3D: Denoising Skeleton for Action Recognition in the dark", which is developed based on mmaction2.
## Environment
```
python==3.8.13
torch==1.12.1+cu116
mmaction2==0.24.1
mmdet==2.25.2
mmpose==0.29.0
```
## Data Preprocessing
Please install the required environment before data preprocessing.
We will mainly use python file 'tools/data/skeleton/ntu_pose_extraction.py', please set mmdet_root and mmpose_root correcttly(The default root of mmpose and mmdetection is ./mm_env).
Put the train set and test set in the following format:
```
mmaction2
│───data
│   │───ntu_dark
│   │   |───mapping_table.txt
│   │   |───ntu_dark_test_list.txt
│   │   |───ntu_dark_train_list.txt
│   │   |───rawframes_train
|   |   |   |   Drink
|   |   |   |   Jump
|   |   |   |   ...
│   │   |───rawframes_val
|   |   |   |   0.mp4
|   |   |   |   1.,p4
|   |   |   |   ...
```
Run python file 'tools/data/skeleton/ntu_pose_extraction.py', select your low-light enhancement method at the beginning. The extracted skeleton will be
```
mmaction2
│───data
│   │───ntu_dark_skeletions
│   │   |───test
|   |   |   |   $SKELETON_FILE_NAME.pkl
|   |   |   |   ...
│   │   |───train
|   |   |   |   $SKELETON_FILE_NAME.pkl
|   |   |   |   ...
```
## Train
Parameter setting is in skeleton3d_config.py, we follow the [standard as mmaction2](https://mmaction2.readthedocs.io/en/latest/tutorials/1_config.html).
You may need to pay attention to ann_file_val and ann_file_train for the data.
You can set gpu_ids and videos_per_gpu for the hardware.
```
python tools/train.py configs/skeleton/posec3d/skeleton3d_config.py \
    --work-dir $YOUR_WORK_DIR \
    --validate --seed 0 --deterministic \
```

## Test
Please put and unzip the test file in ./data/$YOUR_FOLDER
For example, put and unzip data/ntu_test/EE6222test.zip.
```
mmaction2
│───data
│   │───ntu_test
|   |   |   test.txt
│   │   |───test
|   |   |   |   0.mp4
|   |   |   |   1.,p4
|   |   |   |   ...
```
### Data Preprocessing
```
python tools/data/skeleton/ntu_pose_extraction.py
```
Make sure to run val_ntu_skeleton function. After running, test skeleton file ntu_skeleton_GID_0.5_0.5.pkl will be in ./data/ntu_test/

### Model Inference
```
python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
    $MODEL_CHECKPOINT --eval top_k_accuracy mean_class_accuracy
```
The result will be in output.json.txt