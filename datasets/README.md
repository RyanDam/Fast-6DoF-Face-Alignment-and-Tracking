# Prepare the dataset

## General datastruct

Raw data folder struct:
```
- train
    - img1.jpeg
    - img1_ldmks.txt
    ...
    - imgn.jpeg
    - imgn_ldmks.txt
- val
    - img1.jpeg
    - img1_ldmks.txt
    ...
    - imgn.jpeg
    - imgn_ldmks.txt
```

Where:
- `.jpeg`: face image or nonface image
- `_ldmks.txt`: face image must have this file. Which contains landmark point coordinates: each line is `x y` coordinate and have 70 lines.

Data use for training process is a text file contains list of all file for train and a text file contains list of all file for validation. Please check `FaceSynthetics` for example.

## Landmark dataset

This project use 3d 68 points of landmark (difference from the original 300W dataset). Please go to [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) to download the dataset (100K one) and extract it to your disk.

Create your dataset yaml file with the following info:

```yaml
base_path: <path-to-face-synthesis-dataset>/dataset_100000
train: <path-to-list-train-text-file.txt>
val: <path-to-list-val-text-file.txt>
test: <path-to-list-test-text-file.txt>
```

note: you can use list train file in `datasets/FaceSynthetics` for reference.

## Face classification dataset

For face classification, use Widerface dataset to generate nonface data.

### Prepare

Prepare your Widerface dataset with the following format:

```
- train
  - images
    - img1.jpeg
    ..
  - labels
    - img1.txt
    ...
- val
  - images
    - img1.jpeg
    ..
  - labels
    - img1.txt
    ...
```

with the `txt` label file contains face normalized bbox information:

```
0 center_x center_y width height
```

examples:

```
0  0.3984375 0.405668358714044 0.17578125 0.21065989847715735
0  0.62841796875 0.05879864636209814 0.1083984375 0.10575296108291032
0  0.8369140625 0.7529610829103215 0.087890625 0.12521150592216582
```

### Modify script

Modify `Widerface/generate_face_cls_data.py`: 
- `data_path`: path to your train folder
- `target_save`: path to save generated face images

### Generate face and nonface data

Run `Widerface/generate_face_cls_data.py` to generate data.
