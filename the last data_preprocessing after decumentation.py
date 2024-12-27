import cv                            # For image reading and preprocessing
import os                            # For file and directory operations
import numpy as np                   # For array manipulations
import pandas as pd                  # For creating and saving CSV files
from PIL import Image                # To handle images in PIL format
from torchvision import transforms   # For image augmentation transformations
from tqdm import tqdm                # To show progress bars for loops

def preprocess_and_save_csv(input_path, output_path, target_size=(32, 32)):
    os.makedirs(output_path, exist_ok=True)                         # Create the output directory if it does not exist
    
    # ========================Define image augmentations
    transform = transforms.Compose([
        transforms.RandomRotation(10),                             # Randomly rotate the image by up to 10 degrees
        transforms.RandomHorizontalFlip(p=0.3),                    # Flip the image horizontally with a 30% probability
        transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Adjust brightness and contrast
    ])
    
    # ====================Initialize lists to store paths and labels for the CSV file
    processed_paths = []
    labels = []
    
    # ========================Collect all image file paths
    image_files = []
    for root, _, files in os.walk(input_path):                    # Traverse the input directory recursively
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image file extensions
                image_files.append(os.path.join(root, file))      # Add the image file path
                label = os.path.basename(root)                    # Extract the label from the folder name
                labels.append(label)                              # Store the label

    # ==============================Process each image
    for idx, img_path in enumerate(tqdm(image_files)):                         # Show progress with tqdm
        try:
            rel_path = os.path.relpath(os.path.dirname(img_path), input_path)  # Relative path for maintaining folder structure
            output_dir = os.path.join(output_path, rel_path)                   # Create corresponding output directory
            os.makedirs(output_dir, exist_ok=True)                             # Ensure the directory exists
            
            image = cv2.imread(img_path)                                       # Read the image using OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                     # Convert BGR (OpenCV) to RGB (PIL format)
            pil_img = Image.fromarray(image)                                   # Convert the array to PIL Image
            
            pil_img = transform(pil_img)                                       # Apply augmentations
            processed = np.array(pil_img)                                      # Convert back to NumPy array for further processing
            processed = cv2.GaussianBlur(processed, (3, 3), 0)                 # Apply Gaussian blur to the image
            processed = cv2.medianBlur(processed, 3)                           # Apply median blur to reduce noise
            processed = cv2.resize(processed, target_size)                     # Resize the image to the target size
            
            output_file = os.path.join(output_dir, os.path.basename(img_path))      # Define the output file path
            cv2.imwrite(output_file, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))    # Save the processed image
            
            processed_paths.append(os.path.relpath(output_file, output_path))       # Store relative output path
            
        except Exception as e:                                                      # Catch any errors during processing
            print(f"Error processing {img_path}: {str(e)}")
            processed_paths.append(None)                                            # Mark failed processing as None
    
    # Create a DataFrame and save the CSV file
    df = pd.DataFrame({
        'image_path': processed_paths,                                  # Column for processed image paths
        'label': labels                                                 # Column for labels
    })
    
    csv_path = os.path.join(output_path, 'dataset_info.csv')             # Define CSV file path
    df.to_csv(csv_path, index=False)                                     # Save the CSV without the index
    print(f"CSV file saved to: {csv_path}")                              # Print confirmation message

if _name_ == "_main_":
    input_path = r"your\input\path\here"                                 # Specify the input directory
    output_path = r"your\output\path\here"                               # Specify the output directory
    preprocess_and_save_csv(input_path, output_path)                     # Run the preprocessing and CSV creation