python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
    work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.02/best_top1_acc_epoch_640.pth --eval top_k_accuracy mean_class_accuracy

python tools/test.py configs/skeleton/posec3d/skeleton3d_config_test_ntu.py \
    work_dirs/skeleton3dpretrain/GID_0.2_0.5_good_linear_x+_lr_0.02/best_top1_acc_epoch_640.pth --eval top_k_accuracy mean_class_accuracy

# python tools/test.py configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID.py \
#     work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint_GID/best_top1_acc_epoch_940.pth --eval top_k_accuracy mean_class_accuracy

# python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
#     work_dirs/skeleton3dnew/GID_linear_x/best_top1_acc_epoch_925.pth --eval top_k_accuracy mean_class_accuracy

# python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
#     work_dirs/skeleton3dmore/GID_0.2_linear_x/best_top1_acc_epoch_840.pth --eval top_k_accuracy mean_class_accuracy

# python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
#     work_dirs/posec3d/skeleton3d_linear_rl_GID/best_top1_acc_epoch_765.pth --eval top_k_accuracy mean_class_accuracy

# python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
#     work_dirs/posec3d/skeleton3d_linear_rl_x_GID/best_top1_acc_epoch_960.pth --eval top_k_accuracy mean_class_accuracy


# python tools/test.py configs/skeleton/posec3d/skeleton3d_config.py \
#     work_dirs/posec3d/skeleton3d_linear_rl_x_GID/best_top1_acc_epoch_960.pth --eval top_k_accuracy mean_class_accuracy
