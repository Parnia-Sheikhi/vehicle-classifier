import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

class_names = ['SUV', 'bus', 'family sedan', 'fire engine', 'heavy truck',
               'jeep', 'minibus', 'racing car', 'taxi', 'truck']

# same normalization values everywhere, so predictions match what the model
# was actually trained on
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])


def load_trained_model(weights_path="vehicle_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_image(model, device, pil_image):
    image = pil_image.convert("RGB")
    image_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100
    return predicted_class, confidence