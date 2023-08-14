# Face classifier data generator on Wider dataset

## 1. Prepare

Prepare your Wider dataset with the following format:
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

## 2 . Modify script

Modify `generate_face_cls_data.py`: 
- `data_path`: path to your train folder
- `target_save`: path to save generated face images

