import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load MobileNetV2 model
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)

# Load trained model weights
MODEL_PATH = 'training_logs\\20250323_133957\\best_model_epoch16_f10.926.pth'
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Load state dict into the model
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=True)
else:
    model.load_state_dict(checkpoint, strict=True)

model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function for image prediction (now accepts PIL Image or file path)
def predict(image_input):
    # Check if the input is a PIL.Image object
    if isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')  # Ensure image is in RGB format
    else:
        # Assume it's a file path
        image = Image.open(image_input).convert('RGB')

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move image to the same device as the model

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item() == 1  # Return True if the image has lumps