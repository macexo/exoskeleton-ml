# 2025–26 Deployment and Inference Review

Draft for [issue #22](https://github.com/macexo/exoskeleton-ml/issues/22). This contribution covers Summary and How it works. Advantages, disadvantages, open questions, training metrics, and pair review are still pending.

## Summary

Last year's deployment code exports a trained PyTorch Temporal Convolutional Network (TCN) to ONNX and runs it on the Raspberry Pi's CPU. Two separate scripts accept sensor history and return the latest four joint-moment predictions: an [HTTP server](../../archive/2025-26/scripts/server.py) and a [serial endpoint](../../archive/2025-26/scripts/endpoint_tcn.py). They stop at returning predictions; conversion to physical torque and motor actuation are described in the [draft Controls interface](../../archive/2025-26/docs/ml_control_interface.md), but are not implemented in either script.

## How it works

### Export the checkpoint

[`export_onnx.py`](../../archive/2025-26/scripts/export_onnx.py) loads `best_model.pt` and its adjacent `config.yaml`, rebuilds the model, restores its weights, and switches it to evaluation mode. It creates an example input shaped `(1, effective_history, 28)`, exports the model with `torch.onnx.export(..., dynamo=True)`, and saves it to `outputs/model.onnx` by default.

The same script checks the ONNX graph and compares PyTorch and ONNX Runtime outputs for that one random input, using `rtol=1e-3` and `atol=1e-4`. A mismatch prints a warning but leaves the exported file in place. This checks conversion consistency, not prediction accuracy on test data. The script also prints an SCP command for copying the model to the Pi.

### Prepare the sensor data

Each timestep has 28 features: acceleration XYZ and gyroscope XYZ from four IMUs, followed by four joint angles. The configured IMU order is left thigh, left shank, right thigh, right shank; the configured angle order is `[hip_l, hip_r, knee_l, knee_r]`. See [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml).

The [dataset loader](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py) can normalize both inputs and targets using training statistics. Neither deployment script loads those statistics or reverses target normalization. For a checkpoint trained with normalized targets, its raw predictions therefore need an inverse transform before they represent Nm/kg, even though the HTTP response labels them with that unit. The [default model configuration](../../archive/2025-26/configs/model/tcn.yaml) leaves input normalization to the dataset.

### HTTP: send a full window with each request

[`server.py`](../../archive/2025-26/scripts/server.py) loads the ONNX model once at startup with `CPUExecutionProvider`. The [deployment proposal](../../archive/2025-26/docs/deployment_options.md#local-python-server) places sensor collection in a C++ process that calls this local Python service.

1. The caller sends a complete **187 × 28** sensor window. History must be maintained by the caller; the server retains no sensor buffer.
2. `POST /predict` accepts JSON containing a `window` array and validates its dimensions. `POST /predict_msgpack` instead accepts a flat MessagePack array of 5,236 values and reshapes it.
3. The server converts the window to `float32` with shape `(1, 187, 28)` and runs ONNX inference.
4. It takes only the newest timestep, `outputs[0][0, -1, :]`, and returns `hip_left`, `hip_right`, `knee_left`, and `knee_right`.

The response also contains `inference_ms`, which times only `session.run()`. It excludes transport, input validation, and response construction. `GET /health` reports whether the model session is loaded. These routes and their timing logic are all in [`server.py`](../../archive/2025-26/scripts/server.py).

### Serial: keep the window in Python

[`endpoint_tcn.py`](../../archive/2025-26/scripts/endpoint_tcn.py) runs independently of the HTTP server. It loads `outputs/model.onnx` and opens `/dev/ttyACM0` at 115,200 baud.

1. It fills a `deque(maxlen=100)` with zero samples.
2. It reads a newline-terminated message containing 28 comma-separated floats. Messages with the wrong feature count are skipped.
3. It appends the sample, dropping the oldest, and builds a `(1, 100, 28)` input. Predictions during startup include zero-filled history.
4. It runs ONNX inference, takes the newest timestep's four outputs, and writes them back as a comma-separated line with four decimal places.

The response contains no joint names or unit conversion. The sender must know what each output position means.

The two window sizes need reconciling. The [standard configuration](../../archive/2025-26/configs/model/tcn.yaml) declares 187 samples, and the [dedicated exporter](../../archive/2025-26/scripts/export_onnx.py) uses that value without requesting dynamic input dimensions. A 100-sample serial input cannot be assumed compatible with that export. There is a further discrepancy: the [implemented TCN](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py) has two kernel-7 convolutions per block across five blocks with dilations 1, 2, 4, 8, 16. Those layers imply a full receptive field of `1 + 2 × 6 × (1 + 2 + 4 + 8 + 16) = 373` samples, rather than the configured 187.

### Pass the result to Controls

The [interface draft](../../archive/2025-26/docs/ml_control_interface.md#output-specification) proposes converting Nm/kg to Nm by multiplying by participant mass. If targets were normalized during training, they first need their training mean and standard deviation restored. Neither the [server](../../archive/2025-26/scripts/server.py) nor the [serial endpoint](../../archive/2025-26/scripts/endpoint_tcn.py) performs these steps or commands motors.

Joint order also differs between code and specification. The server maps output positions to `[hip_l, hip_r, knee_l, knee_r]`, matching the targets in [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml). The [interface draft](../../archive/2025-26/docs/ml_control_interface.md) specifies `[hip_l, knee_l, hip_r, knee_r]` for outputs and joint-angle inputs. The serial endpoint sends outputs unchanged. Both input and output order must be agreed with Controls before integration; otherwise the right hip and left knee could be exchanged.

### Check inference and timing

The archive includes several supporting tools:

- [`inference_realtime.py`](../../archive/2025-26/scripts/inference_realtime.py) demonstrates PyTorch inference with an initial warmup, a rolling buffer, optional input normalization, and sample or chunk processing. Its main program streams a dataset trial rather than hardware readings.
- [`sample_inference.py`](../../archive/2025-26/scripts/sample_inference.py) compares PyTorch, ONNX, and traced TorchScript outputs and can benchmark them after warmup. It uses synthetic input and defaults to 100 timesteps.
- [`test_server.py`](../../archive/2025-26/scripts/test_server.py) and [`test_endpoint.cpp`](../../archive/2025-26/scripts/test_endpoint.cpp) send synthetic windows to the HTTP server and compare JSON and MessagePack timing. Despite its name, the C++ client does not test the serial endpoint.

The [interface draft](../../archive/2025-26/docs/ml_control_interface.md#timing--real-time-constraints) targets 100 Hz sampling and less than 5 ms inference latency. The HTTP clients compare request timing against a 10 ms loop budget. Their printed pass result therefore does not establish that the 5 ms target or a full sensor-to-motor deadline is met. No latency measurements or hardware tests were run for this draft.
