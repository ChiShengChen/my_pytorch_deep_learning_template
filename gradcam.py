import torch
from torchvision import models, transforms
from captum.attr import LayerGradCam, LayerAttribution
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Load your pre-trained model weights
model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load('path_to_your_model_weights.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Function to visualize and save Grad-CAM
def visualize_and_save_gradcam(image_path, model, layer, output_dir):
    image = preprocess_image(image_path)
    layer_gc = LayerGradCam(model, layer)
    attr = layer_gc.attribute(image, target=0)  # Assuming target class is 0, change as necessary

    # Visualize the Grad-CAM
    attr = LayerAttribution.interpolate(attr, image.shape[2:])
    attr = attr.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(attr, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.repeat(heatmap[..., np.newaxis], 3, axis=2)
    heatmap = Image.fromarray(heatmap).resize((224, 224), Image.ANTIALIAS)
    
    # Load the original image
    orig_image = Image.open(image_path).resize((224, 224), Image.ANTIALIAS)
    orig_image = np.array(orig_image)
    
    # Overlay the heatmap on the original image
    cam_image = np.array(heatmap) * 0.4 + orig_image * 0.6
    cam_image = Image.fromarray(cam_image.astype('uint8'))

    # Save the resulting image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cam_image.save(output_path)

# Example usage
image_path = 'path_to_your_image.jpg'
layer = model.layer4[2].conv3  # Change the layer as necessary
output_dir = 'gradcam_outputs'
visualize_and_save_gradcam(image_path, model, layer, output_dir)
