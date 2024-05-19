import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import nn
from prenet.resnet import resnet50
from prenet.prenet import PRENet
import re
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_cdict():
    _jet_data = {
        'red':   ((0.00, 0, 0),
                  (0.35, 0.5, 0.5),
                  (0.66, 1, 1),
                  (0.89, 1, 1),
                  (1.00, 0.8, 0.8)),
        'green': ((0.000, 0, 0),
                  (0.125, 0, 0),
                  (0.375, 1, 1),
                  (0.640, 1, 1),
                  (0.910, 0.3, 0.3),
                  (1.000, 0, 0)),
        'blue':  ((0.00, 0.30, 0.30),
                  (0.25, 0.8, 0.8),
                  (0.34, 0.8, 0.8),
                  (0.65, 0, 0),
                  (1.00, 0, 0))
    }
    return _jet_data

# Load your pre-trained model weights
weight_path = "/home/meow/cnfood241/food_code_cn241/outputs/prenet_cn241/model.pth"
model = resnet50(pretrained=False)
model = PRENet(model, 512, classes_num=2000)
state_dict = {}
pretrained = torch.load('/home/meow/cnfood241/food_code_cn241/pretrained_models/resnet50-19c8e357.pth')

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
target = 63

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.ToTensor()
])

transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image.requires_grad = True  # Ensure requires_grad is True
    return image

# Example usage
pit_n = '63n000001.jpg'
image_path = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/' + pit_n
output_dir = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/prenet_heatmap'

test_img = Image.open(image_path)
test_img_data = np.asarray(test_img)

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0)  # the model requires a dummy batch dimension

# Identify the target layer for Grad-CAM
target_layer = model.sconv3  # Adjust this to the correct layer in your model

# Wrapper class to handle the forward pass with label
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x, label=None):
        if label is not None:
            xk1, xk2, xk3, x_concat, xc1, xc2, xc3 = self.model(x, label)
        else:
            xk1, xk2, xk3, x_concat, xc1, xc2, xc3 = self.model(x, x)  # Pass input as dummy label
        return x_concat

wrapped_model = ModelWrapper(model)

# Create the CAM object
cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

# Define the target for the class activation map
targets = [ClassifierOutputTarget(target)]

# Generate the CAM
grayscale_cam = cam(input_tensor=input_img, targets=targets)
grayscale_cam = grayscale_cam[0, :]  # Select the CAM for the first image in the batch

# Convert the input image to the correct format for visualization
rgb_img = np.transpose(transformed_img.cpu().numpy(), (1, 2, 0))
rgb_img = rgb_img - np.min(rgb_img)
rgb_img = rgb_img / np.max(rgb_img)

# Create the CAM overlay on the original image
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save the visualization
output_path = os.path.join(output_dir, 'heat_prenet_' + str(target) + pit_n)
plt.imsave(output_path, visualization)
