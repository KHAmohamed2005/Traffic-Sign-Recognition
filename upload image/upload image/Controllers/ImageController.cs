using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.IO;

namespace upload_image.Controllers
{
    public class ImageController : Controller
    {
        private readonly IWebHostEnvironment _webHostEnvironment;

        public ImageController(IWebHostEnvironment webHostEnvironment)
        {
            _webHostEnvironment = webHostEnvironment;
        }

        [HttpGet]
        public IActionResult UploadImage()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> UploadImage(IFormFile image)
        {
            if (image != null && image.Length > 0)
            {
                try
                {
                    // Create upload folder if it doesn't exist
                    string uploadsFolder = Path.Combine(_webHostEnvironment.WebRootPath, "uploads");
                    Directory.CreateDirectory(uploadsFolder);

                    // Generate a unique filename for the image
                    string uniqueFileName = Guid.NewGuid().ToString() + Path.GetExtension(image.FileName);
                    string filePath = Path.Combine(uploadsFolder, uniqueFileName);

                    // Save the image to the server
                    using (var fileStream = new FileStream(filePath, FileMode.Create))
                    {
                        await image.CopyToAsync(fileStream);
                    }

                    // Path to Python executable and Python script
                    string pythonExePath = @"C:\Users\Eng mo\AppData\Local\Programs\Python\Python311\python.exe";
                    string scriptPath = @"D:\predict from torch.utilsLAST.py";  // Path to your Python script
                    string modelPath = @"D:\GTSRB_model_weights.pth";  // Path to your PyTorch model weights file

                    // Ensure that the file paths are quoted correctly to handle spaces
                    string arguments = $"\"{scriptPath}\" \"{modelPath}\" \"{filePath}\"";

                    var startInfo = new ProcessStartInfo
                    {
                        FileName = pythonExePath, // Path to Python executable
                        Arguments = arguments,    // Arguments to the Python script
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    // Execute the Python script
                    using (var process = new Process { StartInfo = startInfo })
                    {
                        process.Start();

                        string output = await process.StandardOutput.ReadToEndAsync();
                        string error = await process.StandardError.ReadToEndAsync();

                        process.WaitForExit();

                        // Handle Python script output
                        if (!string.IsNullOrEmpty(error))
                        {
                            ViewBag.Message = $"Error while processing the image: {error}";
                            ViewBag.Error = error; // Display Python script error
                        }
                        else
                        {
                            ViewBag.Message = "Image processed successfully!";
                            ViewBag.PythonOutput = output; // Display Python script output
                        }
                    }

                    // Set the uploaded image path for the view
                    ViewBag.ImagePath = $"/uploads/{uniqueFileName}";
                }
                catch (Exception ex)
                {
                    ViewBag.Message = "An error occurred while uploading the image.";
                    ViewBag.Error = ex.Message;  // Catch any error during image upload or processing
                }

                return View();
            }

            // Return an error message if the input is invalid
            ViewBag.Message = "Please upload a valid image.";
            return View();
        }
    }
}
