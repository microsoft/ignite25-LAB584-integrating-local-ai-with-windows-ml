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