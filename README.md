

Working in progess...

# Fast 6DoF Face Alignment and Tracking

This project purpose is to implement Ultra lightweight 6 DoF Face Alignment and Tracking. This project is capable of realtime face tracking for edge device.

## Installation

### Requirements

- torch >= 2.0
- autoalbument >= 1.3.1

### Install

[![PyPI version](https://badge.fury.io/py/fdfat.svg)](https://badge.fury.io/py/fdfat)

```
pip install -U fdfat
```

## Model Zoo
 
Check [here](checkpoint/README.md) for currently best model

## Training

Training usually take 2 step:

- Step 1: Train landmark model with the output of 70 landmark points (or 68 for 300W dataset). After the model saturation, process to the next step.
- Step 2: Take the best trained landmark checkpoint, freeze all the parameter and add face classification head and train. This step will add face classify capacity for the tracking while maintain best landmark regression accuracy

All the step is supported by default.

### Prepare the dataset

Check [here](datasets/README.md) to prepare your datasets

### Start training

```bash
fdfat --data <path-to-your-dataset-yaml> --model LightWeightModel
```

For complete list of parameter, please folow this sample config file: [fdfat/cfg/default.yaml](fdfat/cfg/default.yaml)

## Validation

```bash
fdfat --task val --data <path-to-your-dataset-yaml> --model LightWeightModel
```

## Predict

```bash
fdfat --task predict --model LightWeightModel --checkpoint <path-to-checkoint> --input <path-to-test-img>
```

## Export

```bash
fdfat --task export --model LightWeightModel --checkpoint <path-to-checkoint> --export_format tflite
```

## Credit

- [YOLOv8](https://github.com/ultralytics/ultralytics) : Thanks for ultralytics awesome project, I borrow some code from here.
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) : Thanks for your lightweight face detector
- [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) : Thanks for expressive face landmark dataset, it's a good starting point
- [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation) : Thanks for head pose estimation code