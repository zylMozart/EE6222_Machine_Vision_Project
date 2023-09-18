# python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID.py \
#     --validate --seed 0 --deterministic 

python tools/train.py configs/skeleton/posec3d/skeleton3d_config_pretrain.py \
    --validate --seed 0 --deterministic 

# python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py \
#     --validate --seed 0 --deterministic 

# python tools/test.py configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py \
#     work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint2/best_top1_acc_epoch_980.pth \
#     --eval top_k_accuracy