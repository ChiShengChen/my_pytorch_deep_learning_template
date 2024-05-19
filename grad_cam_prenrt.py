import torch
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import nn
import re
import os
from prenet.resnet import resnet50
from prenet.prenet import PRENet

# Load your pre-trained model weights
weight_path = "/home/meow/cnfood241/food_code_cn241/outputs/prenet_cn241/model.pth"
model = resnet50(pretrained=False)
model = PRENet(model, 512, classes_num=2000)
state_dict = {}
pretrained = torch.load('pretrained_models/resnet50-19c8e357.pth')
for k, v in model.state_dict().items():
    if k[9:] in pretrained.keys() and "fc" not in k:
        state_dict[k] = pretrained[k[9:]]
    elif "xx" in k and re.sub(r'xx[0-9]\.?',".", k[9:]) in pretrained.keys():
        state_dict[k] = pretrained[re.sub(r'xx[0-9]\.?',".", k[9:])]
    else:
        state_dict[k] = v
model.load_state_dict(state_dict)
model.fc = nn.Linear(2048, 241)
model.load_state_dict(torch.load(weight_path))
model.eval()
print(model)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image.requires_grad = True  # Ensure requires_grad is True
    return image

class CustomModelWrapper(torch.nn.Module):
    def __init__(self, model, target_layer):
        super(CustomModelWrapper, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hook to save gradients
        self.target_layer.register_full_backward_hook(save_gradient)

    def forward(self, x, label=None):
        output = self.model(x, label)
        if isinstance(output, tuple):
            return output[0]  # Assuming the first element is the primary output
        return output

# Get the target layer
target_layer = model.conv_block3[1].conv  # Change the layer as necessary
print(f"Target layer: {target_layer}")

# Wrap the model
wrapped_model = CustomModelWrapper(model, target_layer)

# Function to visualize and save Grad-CAM
def visualize_and_save_gradcam(image_path, model, target_layer, output_dir, target_class=194):
    image = preprocess_image(image_path)
    grad_cam = LayerGradCam(model, target_layer)

    # Compute Grad-CAM attribution
    attr = grad_cam.attribute(image, target=target_class, relu_attributions=True)

    # Visualize the Grad-CAM
    attr = LayerAttribution.interpolate(attr, image.shape[2:])
    attr = attr.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(attr, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.repeat(heatmap[..., np.newaxis], 3, axis=2)
    heatmap = Image.fromarray(heatmap).resize((600, 600), Image.ANTIALIAS)
    
    # Load the original image
    orig_image = Image.open(image_path).resize((600, 600), Image.ANTIALIAS)
    orig_image = np.array(orig_image)
    
    # Overlay the heatmap on the original image
    cam_image = np.array(heatmap) * 0.4 + orig_image * 0.6
    cam_image = Image.fromarray(cam_image.astype('uint8'))

    # Save the resulting image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cam_image.save(output_path)

# Example usage
image_path = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/000000.jpg'
output_dir = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/prenet_heatmap'
visualize_and_save_gradcam(image_path, wrapped_model, target_layer, output_dir)
