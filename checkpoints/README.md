# Checkpoints for running demo

Checkpoints for running demo or as a pretrain

## Benchmark

### Accurancy

| Model                          | Dataset        | Input size | NME (all) | NME (<= 30°) | NME (> 30°, <= 40°) | NME (> 40°) |
|--------------------------------|----------------|-----------------|-----------|--------------|---------------------|-------------|
| LWModel_ReLU6_affine_cls       | FaceSynthetics | 96x96px           |  4.5144   |       2.8673 |              3.5917 |      7.3704 |
| LWCSPModel_SiLU_affine_facecls | FaceSynthetics | 96x96px           | 4.0641   |        2.535 |              3.1827 |      6.7348 |

Note:
- NME: NME inter-ocular (%)
- All nme is splited to head pose segments to better understanding about head pose challenge.

### Size

| Model                          | Face Classify | Input size | # Params       | GFlop     | Model Size (onnx) |
|--------------------------------|---------------|-----------------|----------------|-----------|-------------------|
| LWModel_ReLU6_affine_cls       | ✅            | 96x96px           | 429K           | 0.3      | 1.7MB           |
| LWCSPModel_SiLU_affine_facecls | ✅            | 96x96px           | 482K           | 0.6      | 1.9MB           |

### Speed

| Model                          | Input size | Apple M1 Max | Intel i5 13600K | Raspberry Pi 4B |
|--------------------------------|-----------------|--------------|-----------------|-----------------|
| LWModel_ReLU6_affine_cls       | 96x96px           | 2.35ms (?)   | 1.07ms          | 16.81ms         |
| LWCSPModel_SiLU_affine_facecls | 96x96px           | 1.7ms        | 1.78ms          | 28.84ms         |

Note:
- Apple M1 Max: 10 core, 32GB ram
- Intel i5 13600K: 16 core, 32GB ram
- Raspberry Pi 4B: 4 core, 8GB ram
- All benchmark using Onnxruntime, 1 thread, 1 process config
- You can run tracking demo with `track_verbose=true` to get your own number

## Preproduce model

- Step 1: train model with the config in `config_landmark.yaml`
- Step 2: took best model from step 1 and train with the config in `config_facecls.yaml`. Basically freezing landmark backbone and add face classification head.
