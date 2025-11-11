## Lab Overview

In this lab demo, we're going to be building an image classification app that can take in any image and locally identify what prominent features might be in the image, like the breed of a dog. We'll be using the ONNX Runtime that ships with WinML, along with an ONNX model we have, and using WinML to dynamically download the EPs for the device.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/18c8ff9f-82bb-41c1-8b12-14c3f5a49af3" />

## Source code

The source code for this session can be found in this [💻 Ignite Lab 584](https://aka.ms/WinML_Lab) repo.

# Introduction

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

### WindowsML (WinML)
Windows Machine Learning (ML) enables C#, C++, and Python developers to run ONNX AI models locally on Windows PCs via the ONNX Runtime, with automatic execution provider management for different hardware (CPUs, GPUs, NPUs). ONNX Runtime can be used with models from PyTorch, Tensorflow/Keras, TFLite, scikit-learn, and other frameworks. Windows ML provides a shared Windows-wide copy of the ONNX Runtime, plus the ability to dynamically download execution providers (EPs).

#### Key benefits
- Dynamically get latest EPs - Automatically downloads and manages the latest hardware-specific execution providers
- Shared ONNX Runtime - Uses system-wide runtime instead of bundling your own, reducing app size
- Smaller downloads/installs - No need to carry large EPs and the ONNX Runtime in your app
- Broad hardware support - Runs on all Windows 11 PCs (x64 and ARM64) with any hardware configuration

# Windows ML Lab Demo

## Step 1: Open the solution

Double click the WinMLLabDemo.sln file in the root directory to open the solution.

<img width="158" height="73" alt="image" src="https://github.com/user-attachments/assets/b2b1787e-e13d-4048-8fe5-0e761ae5e978" /> 

## Step 2: Inspect NuGet packages

In Visual Studio, open the *Solution Explorer* and inspect the dependencies of the project. WindowsAppSDK nuget should already be installed but if you dont see that, right click on the solution and click "Restore Nuget Packages". 

<img width="250" height="144" alt="image" src="https://github.com/user-attachments/assets/590e3c54-d7b6-406e-b76d-b9a4860265d4" />

## Step 3: Deploy the app

Click the Start Debugging button to deploy the app. We'll keep it open while we edit, and see changes appear live!

<img width="269" height="39" alt="image" src="https://github.com/user-attachments/assets/56aeb74f-8efc-420b-9753-3f4a83a041f9" />

The app should look like this when it launches.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/153119fc-1faf-4a61-91b1-7655b34a4963" />

Notice that there are some execution providers that already appear. By default, the CPU and DirectML execution providers are present on all devices. You might have the device with NPU and We're going to use WinML to dynamically download the execution provider that works with your NPU, so that you can run the model on your NPU!


## Step 4: Open the ExecutionLogic.cs file

Further down in the *Solution Explorer*, find and open the **ExecutionLogic.cs** file. Notice that we have a static `OrtEnv` initialized and we have default ONNX code for getting the currently available EPs, but there are a variety of `// TODO`'s for WinML-specific logic we'll implement.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/4fd023ae-9418-40cb-81c3-e5ab7346a0fc" />

## Step 5: Implement getting new EPs

First, we have to use WinML to see if there are any new EPs, and download them if there are. Update `InitializeWinMLEPsAsync` to call `await catalog.EnsureAndRegisterCertifiedAsync()`.

```csharp
public static async Task InitializeWinMLEPsAsync()
{
    // TODO-1: Get/Initialize execution providers from the WinML
    // After finishing this step, WinML will find all applicable EPs for your device
    // download the EP for your device, deploy it and register with ONNX Runtime.

    // Get the WinML EP catalog
    var catalog = ExecutionProviderCatalog.GetDefault();

    // Check if there's any new EPs to download, and if so, download them,
    // and then register all the EPs with the WinML copy of ONNX Runtime
    await catalog.EnsureAndRegisterCertifiedAsync();
}
```

With that method implemented, save your changes (`Ctrl+S`) and then press the **Hot Reload** button (or `Alt+F10`).

<img width="135" height="49" alt="image" src="https://github.com/user-attachments/assets/ff0bb80e-f133-4a23-b899-672e69588351" />

> If you get a hot reload error about "Value cannot be null. (Parameter 'key')", click "Edit" then try adding the first line by itself and hot reloading, and then adding the second line (or stop debugging and re-deploy).

Then, switch back to the app and click the **Initialize WinML EPs** button, which will call the API we just added! The device you're using has NPU and you should see compatible EP in the list.

<img width="359" height="116" alt="image" src="https://github.com/user-attachments/assets/7c6d7342-d261-4ed0-8683-873e2cf5445c" /> <img width="350" height="200" alt="image" src="https://github.com/user-attachments/assets/9ae972f9-893d-4444-a61e-bb6b0e10482c" />


We still need to implement logic to compile, load, and inference the model, which we'll do in the next steps.

## Step 6: Implement compiling the model

For these hardware-specific EPs, models need to be compiled against the EP before you can use the model. 

Back in our **ExecutionLogic.cs** file, locate the `CompileModelForExecutionProvider` method. 

Within the `// TODO-2` in `CompileModelForExecutionProvider`, 

- You'll first need to create a new `compileOptions` via `new OrtModelCompilationOptions(sessionOptions)`, passing in the sessionOptions (created via helper `GetSessionOptions`) that are specific to the EP we've selected.

- Then, you'll need to use `SetInputModelPath` and `SetOutputModelPath` to indicate the source and target model paths.

- Finally, you'll call `CompileModel` to produce the compiled model.

Your final code within `CompileModelForExecutionProvider` should look something like this...

```csharp
var sessionOptions = GetSessionOptions(executionProvider);

// TODO-2: Create compilation options, set the input and output, and compile.
// After finishing this step, a compiled model will be created at 'compiledModelPath'

// Create compilation options from session options
var compileOptions = new OrtModelCompilationOptions(sessionOptions);

// Set input and output model paths
compileOptions.SetInputModelPath(baseModelPath);
compileOptions.SetOutputModelPath(compiledModelPath);

// Compile the model
compileOptions.CompileModel();
```

Save your changes (`Ctrl+S`) and then press the **Hot Reload** button (or `Alt+F10`).

Then, switch back to the app, select the **QNNExecutionProvider**/**OpenVINOExecutionProvider** EP, and click the **Compile Model** button. This will take ~15 seconds, but in the console output you should eventually see that it outputs a compiled model path!

<img width="400" alt="image" src="https://github.com/user-attachments/assets/71c02862-09d6-4891-a55e-0476a1603a15" />

The model is now ready to load on the NPU! Note that our app implements caching logic, so that if the model is already compiled on disk, it will use the already-compiled version, so users only have to experience the initial compile once.

## Step 7: Implement loading the model

Back in our **ExecutionLogic.cs** file, locate the `LoadModel` method. This uses the same `GetSessionOptions` helper method to get sessionOptions, but now instead of compiling a model, we need to load the compiled model.

Our method already passes in the compiled model path, so all we have to do is return a new `InferenceSession` with the compiled model path and the session options! Your completed method should look like...

```csharp
public static InferenceSession LoadModel(string compiledModelPath, OrtEpDevice executionProvider)
{
    var sessionOptions = GetSessionOptions(executionProvider);

    // TODO-3: Return an inference session
    // Return an inference session
    return new InferenceSession(compiledModelPath, sessionOptions);
}
```

Save your changes (`Ctrl+S`), press the **Hot Reload** button (or `Alt+F10`), and switch back to the app, and click the **Load Model** button. You should see console output indicating that the model is loaded!

<img width="400" alt="image" src="https://github.com/user-attachments/assets/9254a1db-78e7-4036-be5c-2f97534a6b4c" />

## Step 8: Implement inferencing the model

Now that our model has been compiled and loaded, we can inference the model!

Back in our **ExecutionLogic.cs** file, locate the `RunModelAsync` method. This method is using the **InferenceSession** we created from the previous step.

We already have a helper method that formats the image inputs into the correct inputs that ONNX expects for this specific model (this code will be different for most models).

We then need to call `session.Run(inputs)`, passing in those inputs, and getting back the results.

The results from ONNX will similarly vary depending on the model, so we have another helper method that we'll use to format those results into text that can be displayed. Use `ModelHelpers.FormatResults(results, session);` to get a display-friendly string from the session run results.

Your final `RunModelAsync` method should look something like this...

```csharp
public static async Task<string> RunModelAsync(InferenceSession session, string imagePath, string compiledModelPath, OrtEpDevice executionProvider)
{
    // Prepare inputs
    var inputs = await ModelHelpers.BindInputs(imagePath, session);

    // TODO-4: Run the inference, format and return the results
    // Run inference
    using var results = session.Run(inputs);

    // Format the results
    return ModelHelpers.FormatResults(results, session);
}
```

Save your changes (`Ctrl+S`), press the **Hot Reload** button (or `Alt+F10`), and switch back to the app, and click the **Run Classification** button. You should see results displayed within the Classification Results text field!

<img width="303" height="257" alt="image" src="https://github.com/user-attachments/assets/b536c26d-d9cc-4f3b-91c1-f1dea16d615e" />

You've successfully completed the lab! We used WinML to get EPs specific to our current device, so that our app didn't have to distribute those EPs ourselves. And then we used the shared copy of ONNX Runtime within WinML to compile, load, and inference this model on NPU!

## Step 9: Experiment with other images or EPs

Feel free to experiment with other images. Click the **Browse** button in the top right and there should be an `image2` image you can select, and then you can run the classification again.

Also, feel free to experiment with using the built-in EPs. Click the **CPUExecutionProvider** or the **DmlExecutionProvider** and then click the **Load Model** button (notice that compiling the model isn't necessary for those), and then click **Run Classification**.

# References
[Windows ML Overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)

[Windows ML API Reference](https://learn.microsoft.com/en-us/windows/windows-app-sdk/api/winrt/microsoft.windows.ai.machinelearning?view=windows-app-sdk-1.8)

