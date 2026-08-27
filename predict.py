import torch
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
import torch.nn as nn
from PIL import Image

class_names = ['SUV', 'bus', 'family sedan', 'fire engine', 'heavy truck',
               'jeep', 'minibus', 'racing car', 'taxi', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("vehicle_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

image_path = "test_image.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]

print(f"Predicted class: {predicted_class}")