# Introduction

### [WindowsML (WinML)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
Windows Machine Learning (ML) enables C#, C++, and Python developers to run ONNX AI models locally on Windows PCs via the ONNX Runtime, with automatic execution provider management for different hardware (CPUs, GPUs, NPUs). ONNX Runtime can be used with models from PyTorch, Tensorflow/Keras, TFLite, scikit-learn, and other frameworks. Windows ML provides a shared Windows-wide copy of the ONNX Runtime, plus the ability to dynamically download execution providers (EPs).

#### Key benefits
- Dynamically get latest EPs - Automatically downloads and manages the latest hardware-specific execution providers
- Shared ONNX Runtime - Uses system-wide runtime instead of bundling your own, reducing app size
- Smaller downloads/installs - No need to carry large EPs and the ONNX Runtime in your app
- Broad hardware support - Runs on all Windows 11 PCs (x64 and ARM64) with any hardware configuration

# Windows ML Lab Demo

In this lab demo, we're going to be building an image classification app that can take in any image and locally identify what prominent features might be in the image, like the breed of a dog. We'll be using the ONNX Runtime that ships with WinML, along with an ONNX model we have, and using WinML to dynamically download the EPs for the device.

<img width="1412" height="961" alt="image" src="https://github.com/user-attachments/assets/18c8ff9f-82bb-41c1-8b12-14c3f5a49af3" />

### ONNX
ONNX (Open Neural Network Exchange). is an open standard for representing machine learning models. It stores the computation graph — the operators and their connections — and the trained weights. The same ONNX file can run on different platforms and hardware without changes.

You can visualize onnx files on <https://netron.app/>

Here's part of Squeezenet model,

<img width="550" height="725" alt="image" src="https://github.com/user-attachments/assets/f3902f50-10ea-403c-9117-0d0ddf9d0491" />

### ONNX Runtime (ORT)
ONNX Runtime, or ORT, is an open‑source engine for running ONNX models. It loads the model graph and weights, executes the operators, and returns the output.

### Execution Provider (EP)
Execution providers (EPs) are specialized implementations that execute ONNX operations on specific hardware. They act as the interface between ONNX Runtime and hardware-specific libraries to leverage hardware acceleration capabilities. IOW EPs are plug-ins that tell ONNX Runtime where and how to run your model.

### Model compilation
Compilation applies graph-level optimizations (e.g. fusion, constant folding, memory layout changes etc.) and hardware-specific tuning.
E.g. for QNN EP - Compilation step maps ONNX ops → QNN ops. IOW it compiles the ONNX graph into a QNN graph.

### Model loading
Some of the steps which are part of model loading,
- Read .onnx file from disk or memory
- Parse protobuf into internal model structure
- Validate opset, schema, and available operators

