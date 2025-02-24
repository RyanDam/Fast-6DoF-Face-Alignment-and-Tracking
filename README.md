# Fast 6DoF Face Alignment and Tracking

![Demo](media/300VW_Trainset_009_processed.gif)

Facial alignment and Face tracking is no news but gain significant usage recently. The wave of  application such as argumented reality or virtual reality, self driving car or even face beauty app on mobile phone is rising recently year. All that application have the same aspect: edge device with low computaion capacity or run on battery. To optimize speed and efficiency on that devices, researcher and engineer ten to limits the capacity of the AI model results in decrease accurancy. This paper introduce a novel, efficency and robust way to track Facial feature, which will run only on cpu and easy to adap to other subject such as human, hand, etc...

This project purpose is to implement Ultra lightweight 6 DoF Face Alignment and Tracking. This project is capable of realtime face tracking for edge device using only CPU.

## Installation

### Requirements

- torch >= 2.0
- autoalbument >= 1.3.1

### Install

[![PyPI version](https://badge.fury.io/py/fdfat.svg)](https://badge.fury.io/py/fdfat)

```
pip install -U fdfat
```

## Demo

For demo, you need face detector model [version-RFB-320.onnx](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/models/onnx)

Running using webcam

```
fdfat \
--task track \
--track_detector <path-to-version-RFB-320.onnx> \
--track_landmark <path-to-landmark-model-onnx> \
--track_source camera
```

Running using video

```
fdfat \
--task track \
--track_detector <path-to-version-RFB-320.onnx> \
--track_landmark <path-to-landmark-model-onnx> \
--input media/300VW_Trainset_009.mp4 \
```

## Model Zoo
 
| Model                          | Input size | Apple M1 Max | Intel i5 13600K | Raspberry Pi 4B |
|--------------------------------|-----------------|--------------|-----------------|-----------------|
| LWModel_ReLU6_affine_cls       | 96x96px           | 2.35ms (?)   | 1.07ms          | 16.81ms         |
| LWCSPModel_SiLU_affine_facecls | 96x96px           | 1.7ms        | 1.78ms          | 28.84ms         |

| Model                          | Dataset        | Input size | NME (all) | NME (<= 30°) | NME (> 30°, <= 40°) | NME (> 40°) |
|--------------------------------|----------------|-----------------|-----------|--------------|---------------------|-------------|
| LWModel_ReLU6_affine_cls       | FaceSynthetics | 96x96px           |  4.5144   |       2.8673 |              3.5917 |      7.3704 |
| LWCSPModel_SiLU_affine_facecls | FaceSynthetics | 96x96px           | 4.0641   |        2.535 |              3.1827 |      6.7348 |

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
fdfat --data <path-to-your-dataset-yaml> --model LWModel
```

For complete list of parameter, please folow this sample config file: [fdfat/cfg/default.yaml](fdfat/cfg/default.yaml)

## Validation

```bash
fdfat --task val --data <path-to-your-dataset-yaml> --model LWModel
```

## Predict

```bash
fdfat --task predict --model LWModel --checkpoint <path-to-checkoint> --input <path-to-test-img>
```

## Export

```bash
fdfat --task export --model LWModel --checkpoint <path-to-checkoint> --export_format tflite
```

## Credit

- [YOLOv8](https://github.com/ultralytics/ultralytics) : Thanks for ultralytics awesome project, I borrow some code from here.
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) : Thanks for your lightweight face detector
- [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) : Thanks for expressive face landmark dataset, it's a good starting point
- [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation) : Thanks for head pose estimation code
