import torch
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import nn
from timm.models import create_model
import captum
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

def get_cdict():
    _jet_data = {
              # 'red':   ((0.00, 0, 0),
              #          (0.35, 0, 0),
              #          (0.66, 1, 1),
              #          (0.89, 1, 1),
              #          (1.00, 0.5, 0.5)),
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
             # 'blue':  ((0.00, 0.5, 0.5),
             #           (0.11, 1, 1),
             #           (0.34, 1, 1),
             #           (0.65, 0, 0),
             #           (1.00, 0, 0))}
             'blue':  ((0.00, 0.30, 0.30),
                       (0.25, 0.8, 0.8),
                       (0.34, 0.8, 0.8),
                       (0.65, 0, 0),
                       (1.00, 0, 0))
             }
    return _jet_data

# Load your pre-trained model weights
weight_path = "/home/meow/cnfood241/food_code_cn241/outputs/tf_efficientnet_b6.ap_in1k_cnf241/model.pth"
model = create_model('tf_efficientnet_b6.ap_in1k', pretrained=True, num_classes=241)
model.load_state_dict(torch.load(weight_path))
model.eval()

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
output_dir = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/caml_heatmap'

test_img = Image.open(image_path)
test_img_data = np.asarray(test_img)

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0)  # the model requires a dummy batch dimension

# Identify the target layer for Grad-CAM
target_layer = model.blocks[-1]  # Adjust this to the correct layer in your model

layer_gradcam = LayerGradCam(model, target_layer)
attributions_lgc = layer_gradcam.attribute(input_img, target=target)

upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])
print(upsamp_attr_lgc[0].shape)

print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

# Debug: Check for zero scale factor
attr_np = upsamp_attr_lgc[0].cpu().permute(1, 2, 0).detach().numpy()
if np.max(attr_np) == np.min(attr_np):
    print("Warning: Attributions have zero scale factor. Normalization will fail.")


integrated_gradients = IntegratedGradients(model)
print("here")
attributions_ig = integrated_gradients.attribute(input_img, target=21, n_steps=200)


cdict = get_cdict()
cmap = matplotlib.colors.LinearSegmentedColormap("jet_revice", cdict)

fig, ax =plt.subplots()
_ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                method='blended_heat_map',
                cmap=cmap,
                show_colorbar=False,
                sign='positive',
                outlier_perc=1,
                plt_fig_axis=(fig, ax)
                # title='Blended Heat Map'
)

# Save the figure
output_path = os.path.join(output_dir, 'heat_caml_' + str(target) + pit_n)
fig.savefig(output_path)