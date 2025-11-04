using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using Microsoft.Windows.AI.MachineLearning;
using Microsoft.Windows.AppNotifications;
using Microsoft.Windows.AppNotifications.Builder;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Xml.Linq;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Streams;
using IOPath = System.IO.Path;

namespace WinMLLabDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private InferenceSession? _loadedSession;
        public ObservableCollection<OrtEpDevice> ExecutionProviders { get; set; }
        private string selectedImagePath = string.Empty;
        private OrtEpDevice? selectedExecutionProvider = null;
        private const string ModelName = "SqueezeNet";
        private const string ModelExtension = ".onnx";

        public MainWindow()
        {
            InitializeComponent();

            ExecutionProviders = new ObservableCollection<OrtEpDevice>();
            ExecutionProvidersGrid.ItemsSource = ExecutionProviders;
            
            // Set up EP selection event
            ExecutionProvidersGrid.SelectionChanged += ExecutionProvidersGrid_SelectionChanged;
            
            // Initialize with some sample data
            LoadExecutionProviders();
            WriteToConsole("WinML Demo Application initialized.");

            // Select the default image
            SelectImage(IOPath.Combine(AppDomain.CurrentDomain.BaseDirectory, "image.jpg"));
        }

        private void LoadExecutionProviders()
        {
            ExecutionProviders.Clear();

            var eps = ExecutionLogic.LoadExecutionProviders();

            foreach (var ep in eps)
            {
                ExecutionProviders.Add(ep);
            }

            WriteToConsole("Loaded execution providers.");
        }

        private void ExecutionProvidersGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ExecutionProvidersGrid.SelectedItem is OrtEpDevice selectedEP)
            {
                selectedExecutionProvider = selectedEP;
                _loadedSession = null;
                WriteToConsole($"Selected execution provider: {selectedEP.EpName}");
            }
            else
            {
                selectedExecutionProvider = null;
                _loadedSession = null;
            }

            // Update button states
            UpdateButtonStates();
        }

        private void UpdateButtonStates()
        {
            if (selectedExecutionProvider == null)
            {
                CompileModelButton.IsEnabled = false;
                LoadModelButton.IsEnabled = false;
                RunButton.IsEnabled = false;
            }

            CompileModelButton.IsEnabled = true;

            string compiledModelPath = ModelHelpers.GetCompiledModelPath(selectedExecutionProvider!);
            LoadModelButton.IsEnabled = File.Exists(compiledModelPath);

            RunButton.IsEnabled = _loadedSession != null;
        }

        private void RefreshEPButton_Click(object sender, RoutedEventArgs e)
        {
            LoadExecutionProviders();
            // Reset selection state
            selectedExecutionProvider = null;
            _loadedSession = null;
            CompileModelButton.IsEnabled = false;
            RunButton.IsEnabled = false;
        }

        private async void InitializeWinMLEPsButton_Click(object sender, RoutedEventArgs e)
        {
            InitializeWinMLEPsButton.IsEnabled = false;
            try
            {
                WriteToConsole("WinML: Downloading and registering EPs...");
                var now = DateTime.Now;

                // Download and register the Execution Providers for our device
                await ExecutionLogic.InitializeWinMLEPsAsync();

                var elapsed = DateTime.Now - now;
                WriteToConsole($"WinML: EPs downloaded and registered in {elapsed.TotalMilliseconds} ms.");
                LoadExecutionProviders();
            }
            catch (Exception ex)
            {
                WriteToConsole($"Error downloading execution providers: {ex.Message}");
            }
            finally
            {
                InitializeWinMLEPsButton.IsEnabled = true;
            }
        }

        private async void CompileModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (selectedExecutionProvider == null)
            {
                WriteToConsole("No execution provider selected.");
                return;
            }

            switch (selectedExecutionProvider.EpName)
            {
                case "CPUExecutionProvider":
                case "DmlExecutionProvider":
                    WriteToConsole("Compiling isn't necessary for CPU or DML EPs");
                    return;
            }

            CompileModelButton.IsEnabled = false;
            try
            {
                WriteToConsole($"Compiling model for {selectedExecutionProvider.EpName}...");
                var now = DateTime.Now;

                string compiledModelPath = await Task.Run(() => ExecutionLogic.CompileModelForExecutionProvider(selectedExecutionProvider));

                var elapsed = DateTime.Now - now;
                WriteToConsole($"Model compiled successfully in {elapsed.TotalMilliseconds} ms: {compiledModelPath}");
                
                // Update Run button state
                UpdateButtonStates();
            }
            catch (Exception ex)
            {
                WriteToConsole($"Error compiling model: {ex.Message}");
            }
            finally
            {
                CompileModelButton.IsEnabled = true;
            }
        }

        

        private void BrowseImageButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Title = "Select an image file",
                Filter = "Image files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|All files (*.*)|*.*",
                InitialDirectory = IOPath.Combine(AppDomain.CurrentDomain.BaseDirectory)
            };

            if (openFileDialog.ShowDialog() == true)
            {
                SelectImage(openFileDialog.FileName);
            }
        }

        private void SelectImage(string filePath)
        {
            selectedImagePath = filePath;
            ImagePathTextBox.Text = selectedImagePath;

            try
            {
                // Load and display the selected image
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(selectedImagePath);
                bitmap.DecodePixelWidth = 300; // Limit size for preview
                bitmap.EndInit();
                SelectedImage.Source = bitmap;

                WriteToConsole($"Selected image: {IOPath.GetFileName(selectedImagePath)}");
            }
            catch (Exception ex)
            {
                WriteToConsole($"Error loading image: {ex.Message}");
            }
        }

        private async void RunButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(selectedImagePath))
            {
                WriteToConsole("Please select an image first.");
                return;
            }

            if (selectedExecutionProvider == null)
            {
                WriteToConsole("Please select an execution provider first.");
                return;
            }

            string compiledModelPath = ModelHelpers.GetCompiledModelPath(selectedExecutionProvider);
            if (!File.Exists(compiledModelPath))
            {
                WriteToConsole("Compiled model not found. Please compile the model first.");
                return;
            }

            WriteToConsole($"Running classification on: {IOPath.GetFileName(selectedImagePath)}");
            WriteToConsole($"Using execution provider: {selectedExecutionProvider.EpName}");
            WriteToConsole($"Using compiled model: {IOPath.GetFileName(compiledModelPath)}");
            
            // Disable all buttons during inference
            CompileModelButton.IsEnabled = false;
            LoadModelButton.IsEnabled = false;
            RunButton.IsEnabled = false;
            
            try
            {
                ResultsTextBlock.Text = "Running classification ...";
                DateTime start = DateTime.Now;
                var results = await Task.Run(() => ExecutionLogic.RunModelAsync(_loadedSession, selectedImagePath, compiledModelPath, selectedExecutionProvider));
                ResultsTextBlock.Text = results;
                var time = DateTime.Now - start;
                WriteToConsole($"Classification completed successfully in {time.TotalMilliseconds} ms.");
            }
            catch (Exception ex)
            {
                WriteToConsole($"Error during classification: {ex.Message}");
                ResultsTextBlock.Text = $"Error during classification: {ex.Message}";
            }
            finally
            {
                // Re-enable the button
                CompileModelButton.IsEnabled = true;
                LoadModelButton.IsEnabled = true;
                RunButton.IsEnabled = true;
            }
        }

        private void ClearConsoleButton_Click(object sender, RoutedEventArgs e)
        {
            ConsoleTextBlock.Text = string.Empty;
        }

        public void WriteToConsole(string message)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            var logEntry = $"[{timestamp}] {message}\n";
            
            Dispatcher.Invoke(() =>
            {
                ConsoleTextBlock.Text += logEntry;
                
                // Auto-scroll to bottom
                if (ConsoleTextBlock.Parent is ScrollViewer scrollViewer)
                {
                    scrollViewer.ScrollToEnd();
                }
            });
        }

        private async void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (selectedExecutionProvider == null)
            { 
                WriteToConsole("Please select an execution provider first.");
                return;
            }

            var path = ModelHelpers.GetCompiledModelPath(selectedExecutionProvider);
            if (!File.Exists(path))
            {
                WriteToConsole($"Compiled model not found: {path}");
                return;
            }

            try
            {
                WriteToConsole($"Loading model for execution provider: {selectedExecutionProvider.EpName}");
                DateTime start = DateTime.Now;
                _loadedSession = await Task.Run(() => ExecutionLogic.LoadModel(ModelHelpers.GetCompiledModelPath(selectedExecutionProvider), selectedExecutionProvider));
                var elapsed = DateTime.Now - start;
                WriteToConsole($"Model loaded successfully in {elapsed.TotalMilliseconds} ms.");
                RunButton.IsEnabled = true;
            }
            catch (Exception ex)
            {
                WriteToConsole($"Error loading model: {ex.Message}");
            }
        }
    }
}