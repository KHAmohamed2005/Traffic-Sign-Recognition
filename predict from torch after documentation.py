import torch           #imports the PyTorch library  for building and training deep learning models.
import torch.nn as nn  # imports the nn module from PyTorch,  for creating neural network architectures .
import torchvision.transforms as transforms # imports the transforms module from the torchvision library,  for data augmentation and preprocessing.
from PIL import Image  # imports the Image class from the Pillow (PIL) library, which is used for working with images in Python.
import os              # imports the os module,  for interacting with the operating system, such as navigating directories, creating files, and accessing environment variables.

import os


img_transforms = transforms.Compose([   #This line initializes a sequence of image transformations to be applied during data preprocessing.
    transforms.Resize((32, 32)),        # input images to a fixed dimension of 32 pixels in both width and height.
    transforms.RandomHorizontalFlip(),  #randomly flips the image horizontally during data augmentation
    transforms.ToTensor(),              #converts the image data from a PIL Image or NumPy array to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))# normalizes pixel values to a range of [-1, 1].
])

class GTSRBClassifier(nn.Module):  # This indicates that the class will represent a neural network module.
    def __init__(self):            #where you initialize the attributes of the class.
        super(GTSRBClassifier,self).__init__() #  ensuring that the initialization of the base class is also performed.
        self.model = nn.Sequential(            # to hold a series of layers that will form the core of the GTSRB classifier
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # This line defines a 2D convolutional layer.
            nn.ReLU(), #activation function after the convolutional layer. ReLU introduces non-linearity into the model,
            nn.MaxPool2d(kernel_size=2),#downsamples feature maps using 2x2 max-pooling.

            # second layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # third layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),

            # flatten and apply linear
            nn.Flatten(),                                    #Convert the multi-dimensional feature maps into a single vector.
            nn.Linear(in_features=128*8*8, out_features=256),#Perform a linear transformation from the flattened vector to a 256-dimensional space.
            nn.ReLU(),                                        #: Introduce non-linearity using the ReLU activation function.
            nn.Linear(in_features=256, out_features=43)#Map the 256-dimensional features to the final output layer with 43 neurons, corresponding to the 43 classes.
        )

    def forward(self, x):
        return self.model(x)

def predict_single_image(model, image_path, transform, class_labels):
    """
    Function to predict the class of a single image using a trained model.
    
    Parameters:
        model (nn.Module): The trained model.
        image_path (str): Path to the image file.
        transform (callable): Transformations to apply to the image.
        class_labels (list): List of class labels corresponding to model outputs.
    
    Returns:
        str: Predicted class label or an error message.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Load and preprocess the image
    if not os.path.exists(image_path):
        return f"Error: Image not found at path {image_path}"#This code checks if the specified image path exists. If not, it returns an error message indicating the invalid path.

    try:
        image = Image.open(image_path).convert('RGB')#If the image is opened successfully, it's then converted to RGB format 
    except Exception as e:
        return f"Error loading image: {e}" # returns an error message indicating that there was an issue loading the image
    
    # Transform the image and add a batch dimension
    image = transform(image).unsqueeze(0)

    # this checks for GPU availability and sets the device (device) to either "cuda" (GPU) or "cpu" depending on what's available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Make prediction
    with torch.no_grad():  # Disable gradient calculations for inference
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get class with the highest probability
    
    # Get predicted label
    predicted_label = class_labels[predicted.item()]#corresponding to the possible output indices from the model and extracts the actual integer value from the single-element tensor predicted.
    return f"Predicted Class: {predicted_label}" #constructs and returns a formatted string that includes the predicted class label


# Example usage

# Path to the test image
image_path = r"C:\Users\Mahmoud Salman\Downloads\Telegram Desktop\data (2)\data\Test\00056.png"  # Replace with the path to your test image

# Class labels (43 classes from the dataset)
class_labels = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
    'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 
    'Beware of ice/snow', 'Wild animals crossing', 'End speed + passing limits', 
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
    'End of no passing', 'End no passing veh > 3.5 tons'
]

# Check if the model weights file exists
model_weights_path = r"C:\Users\Mahmoud Salman\Downloads\Telegram Desktop\data (2)\GTSRB_model_weights.pth" #defines the path to the file containing the trained model weights
if not os.path.exists(model_weights_path):#checks if the model weights file exists at the specified path 
    print(f"Error: Model weights file not found at path {model_weights_path}")#his line is executed only if the model weights file is not found. It prints an error message indicating the missing file and its path.
else:
    # Load the trained model
    model = GTSRBClassifier()  # Reinitialize the model architecture
    model.load_state_dict(torch.load(model_weights_path))  # Load the trained weights
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))#This line checks for GPU availability 

    # Predict the class of a single image
    predicted_class = predict_single_image(model, image_path, img_transforms, class_labels)
    print(predicted_class)#the final output of this code block
