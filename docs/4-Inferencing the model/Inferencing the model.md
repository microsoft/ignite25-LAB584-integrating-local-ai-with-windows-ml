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