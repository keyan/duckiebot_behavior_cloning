# data_converter

Uses code from the official [duckietown repo](https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning/tree/master/duckieSchool/duckieRoad) to convert recorded robot runs from ROS bags to a data format suitable for downstream parsing and conversion to useful model training input.

See also: https://docs.duckietown.org/daffy/AIDO/out/embodied_bc.html

## Usage

1. Change `ROBOT_NAME` in `src/extract_data.py`
1. Run:
    ```
    make image
    make extract
    ```
1. If the stdout message indicates some frames were extracted, verify the log file is worth keeping with:
    ```
    python ../log_utils/log_viewer.py --log_name converted/YOUR_LOG.log
    ```

## bag_files
Each ROS bag file has an accompanying gif to show the rough path taken. All bag data is from publically available logs on: http://logs.duckietown.org/.

The actual bag files are not included in this repo because they are very large.

### Bags in use
1. `20171222163246_tori`
    - 107 frames, low/med quality
1. `20160504183139_bill`
    - 2450 frames, med quality, some noisy controls, has stop signs
1. `20160504223118_amadobot`
    - 2521 frames, low quality, good candidate for removal
1. `20160505011520_maserati`
    - 3106 frames, med quality, has stop signs

### Bags attempted
Could not extract any frames:
```
20160314210733_penguin
20180108195947_a313
20171226183955_yaf
20160503185748_starduck
```

Very poor quality data extracted:
```
20171112233301_misteur
```
