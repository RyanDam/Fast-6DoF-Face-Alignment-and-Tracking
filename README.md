Working in progess...

# Fast 6DoF Face Alignment and Tracking

This project purpose is to implement Ultra lightweight 6 DoF Face Alignment and Tracking. This project is capable of realtime tracking face for mobile device.

## Installation

### Requirements

- torch >= 2.0
- autoalbument >= 1.3.1

### Install

```
pip install -U fdfat
```

## Model Zoo

## Training

### Prepare the dataset

This project use 3d 68 points of landmark (difference from the original 300W dataset). Please go to [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) to download the dataset (100K one) and extract it to your disk.

Create your dataset yaml file with the following 

### Start training

## Predict

## Credit

- [YOLOv8](https://github.com/ultralytics/ultralytics) : Thanks for ultralytics awesome project, I borrow some code from here.
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) : Thanks for your lightweight face detector
- [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) : Thanks for expressive face landmark dataset, it's a good starting point
- [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation) : Thanks for head pose estimation code