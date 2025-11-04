using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Streams;
using IOPath = System.IO.Path;

namespace WinMLLabDemo
{
    internal static class ModelHelpers
    {
        private const string ModelName = "SqueezeNet";
        private const string ModelExtension = ".onnx";

        public static string FormatResults(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, InferenceSession session)
        {
            // Extract output tensor
            string outputName = session.OutputMetadata.First().Key;
            float[] resultTensor = results.First(r => r.Name == outputName).AsEnumerable<float>().ToArray();

            // Load labels from deployed app root directory and print results
            string labelsPath = IOPath.Combine(AppDomain.CurrentDomain.BaseDirectory, "SqueezeNet.Labels.txt");
            IList<string> labels = ModelHelpers.LoadLabels(labelsPath);
            return FormatResults(labels, resultTensor);
        }

        private static string FormatResults(IList<string> labels, IReadOnlyList<float> results)
        {
            // Apply softmax to the results
            float maxLogit = results.Max();
            var expScores = results.Select(r => MathF.Exp(r - maxLogit)).ToList(); // stability with maxLogit
            float sumExp = expScores.Sum();
            var softmaxResults = expScores.Select(e => e / sumExp).ToList();

            // Get top 5 results
            IEnumerable<(int Index, float Confidence)> topResults = softmaxResults
                .Select((value, index) => (Index: index, Confidence: value))
                .OrderByDescending(x => x.Confidence)
                .Take(5);

            // Format results
            StringBuilder resultBuilder = new StringBuilder();
            resultBuilder.AppendLine("Top Predictions:");
            resultBuilder.AppendLine("-------------------------------------------");
            // Replace the incorrect AppendLine call with AppendFormat followed by AppendLine
            resultBuilder.AppendFormat("{0,-32} {1,10}", "Label", "Confidence");
            resultBuilder.AppendLine();
            resultBuilder.AppendLine("-------------------------------------------");

            foreach (var result in topResults)
            {
                resultBuilder.AppendFormat("{0,-32} {1,10:P2}", labels[result.Index], result.Confidence);
                resultBuilder.AppendLine();
            }

            resultBuilder.AppendLine("-------------------------------------------");
            return resultBuilder.ToString();
        }

        public static IList<string> LoadLabels(string labelsPath)
        {
            return File.ReadAllLines(labelsPath)
                .Select(line => line.Split(',', 2)[1])
                .ToList();
        }

        public static async Task<List<NamedOnnxValue>> BindInputs(string imagePath, InferenceSession session)
        {
            DenseTensor<float> input = await ModelHelpers.PreprocessImageAsync(imagePath);

            // Prepare input tensor
            string inputName = session.InputMetadata.First().Key;
            DenseTensor<float> inputTensor = new(
                input.ToArray(),
                [1, 3, 224, 224], // Shape of the tensor
                false             // isReversedStride should be explicitly set to false
            );

            // Bind inputs
            List<NamedOnnxValue> inputs = [NamedOnnxValue.CreateFromTensor(inputName, inputTensor)];
            return inputs;
        }

        public static async Task<DenseTensor<float>> PreprocessImageAsync(string imagePath)
        {
            VideoFrame videoFrame = await LoadImageFileAsync(imagePath);
            return await PreprocessImageAsync(videoFrame);
        }

        public static async Task<DenseTensor<float>> PreprocessImageAsync(VideoFrame videoFrame)
        {
            SoftwareBitmap softwareBitmap = videoFrame.SoftwareBitmap;
            const int targetWidth = 224;
            const int targetHeight = 224;

            float[] mean = [0.485f, 0.456f, 0.406f];
            float[] std = [0.229f, 0.224f, 0.225f];

            // Convert to BGRA8
            if (softwareBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 ||
                softwareBitmap.BitmapAlphaMode != BitmapAlphaMode.Premultiplied)
            {
                softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
            }

            // Resize
            softwareBitmap = await ResizeSoftwareBitmapAsync(softwareBitmap, targetWidth, targetHeight);

            // Get pixel data
            uint bufferSize = (uint)(targetWidth * targetHeight * 4);
            Windows.Storage.Streams.Buffer buffer = new(bufferSize);
            softwareBitmap.CopyToBuffer(buffer);
            byte[] pixels = buffer.ToArray();

            // Output Tensor shape: [1, 3, 224, 224]
            DenseTensor<float> tensorData = new([1, 3, targetHeight, targetWidth]);

            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    int pixelIndex = (y * targetWidth + x) * 4;
                    float r = pixels[pixelIndex + 2] / 255f;
                    float g = pixels[pixelIndex + 1] / 255f;
                    float b = pixels[pixelIndex + 0] / 255f;

                    // Normalize using mean/stddev
                    r = (r - mean[0]) / std[0];
                    g = (g - mean[1]) / std[1];
                    b = (b - mean[2]) / std[2];

                    int baseIndex = y * targetWidth + x;
                    tensorData[0, 0, y, x] = r; // R
                    tensorData[0, 1, y, x] = g; // G
                    tensorData[0, 2, y, x] = b; // B
                }
            }

            return tensorData;
        }

        public static async Task<VideoFrame> LoadImageFileAsync(string filePath)
        {
            try
            {
                StorageFile file = await StorageFile.GetFileFromPathAsync(filePath);
                IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read);
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                SoftwareBitmap softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                return inputImage;
            }
            catch (FileNotFoundException)
            {
                throw new FileNotFoundException($"Image file not found: {filePath}");
            }
            catch (Exception ex)
            {
                throw new Exception($"Error loading image: {ex.Message}", ex);
            }
        }

        public static async Task<SoftwareBitmap> ResizeSoftwareBitmapAsync(SoftwareBitmap bitmap, int width, int height)
        {
            using InMemoryRandomAccessStream stream = new();
            BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, stream);
            encoder.SetSoftwareBitmap(bitmap);
            encoder.IsThumbnailGenerated = false;
            await encoder.FlushAsync();
            stream.Seek(0);

            BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
            BitmapTransform transform = new()
            {
                ScaledWidth = (uint)width,
                ScaledHeight = (uint)height,
                InterpolationMode = BitmapInterpolationMode.Fant
            };
            SoftwareBitmap resized = await decoder.GetSoftwareBitmapAsync(
                BitmapPixelFormat.Bgra8,
                BitmapAlphaMode.Premultiplied,
                transform,
                ExifOrientationMode.IgnoreExifOrientation,
                ColorManagementMode.DoNotColorManage);

            stream.Dispose();
            return resized;
        }

        public static string GetCompiledModelPath(OrtEpDevice ep)
        {
            if (ep == null)
            {
                return "";
            }

            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;

            // CPU and DML don't need to be compiled
            switch (ep.EpName)
            {
                case "CPUExecutionProvider":
                case "DmlExecutionProvider":
                    return IOPath.Combine(baseDirectory, $"{ModelName}{ModelExtension}");
            }

            string compiledModelName = $"{ep.EpName}.{ModelName}{ModelExtension}";
            return IOPath.Combine(baseDirectory, compiledModelName);
        }
    }
}
