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
