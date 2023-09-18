for action in data/ntu_dark/rawframes_train/*
do 
    for video in $action/*
    do
        echo $video
    done
    break
done

# python tools/data/skeleton/ntu_pose_extraction.py tools/data/skeleton/S001C001P001R001A001_rgb.avi tools/data/skeleton/S001C001P001R001A001.pkl